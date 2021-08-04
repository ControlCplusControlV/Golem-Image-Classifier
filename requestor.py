#!/usr/bin/env python3
"""
the requestor agent controlling and interacting with Image Classifier
"""
import asyncio
from datetime import datetime, timedelta, timezone
import pathlib
import random
import string
import sys
import time
import argparse
from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast
from yapapi import (
    NoPaymentAccountError,
    __version__ as yapapi_version,
    windows_event_loop_fix,
)
from yapapi import Golem
from yapapi.services import Service, ServiceState

from yapapi.log import enable_default_logger, pluralize
from yapapi.payload import vm


NUM_INSTANCES = 1
STARTING_TIMEOUT = timedelta(minutes=100)

class Store:
    classes = ""
    dataset = ""
GlobalStore = Store()
class ImageClassifierService(Service):
    CLASSIFIER = "/golem/run/ImageClassification.py"
    CLASSIFIERCLIENT = "/golem/run/ClassifierClient.py"
    @staticmethod
    async def get_payload():
        return await vm.repo(
            image_hash="5a5f3664b6f45263dc7c15a1107ad5c0ab0df3b4745c8194a8cbc8dd",
            # For ML these definitely should be tuned, but was on testnet and couldn't be picky
            min_mem_gib=8,
            min_storage_gib=20,
        )

    async def start(self):
        """
        Transfers over model weights, and starts prediction service, in addition to model   
        REQUIRED PARAMS - dataset.tar.gz
        """
        data = "/golem/work/" + GlobalStore.dataset 
        self._ctx.send_file(str("vgg16.h5"), str("/golem/work/vgg16.h5"))
        sent = yield self._ctx.commit()
        finished = await sent
        #Send in the dataset as a zipped file
        self._ctx.send_file(str(GlobalStore.dataset), str(data))
        self._ctx.run("/bin/tar","--no-same-owner", "-C", "/golem/work/", "-xzvf", data)
        # Start Main classification script so http server can forward it requests
        self._ctx.run("/bin/sh", "-c", "nohup python /golem/run/ImageClassification.py run &")
        servicestart = yield self._ctx.commit()
        done1 = await servicestart # This is required or it will execute the next script before socket is created, so delay is neccesary
        trainpath = "/golem/work/" + GlobalStore.dataset + "/train"
        validset = "/golem/work/" + GlobalStore.dataset + "/valid"
        self._ctx.run(self.CLASSIFIERCLIENT,"-t", trainpath, "-v", validset, "--start","-c", GlobalStore.classes)
        built = yield self._ctx.commit()
        done = await built
    async def run(self):
        """
        Starts a quick http server, accepts predict and train requests in packets
        """
        async def predict(imagename):
            imagename = input("What is the name of the image you wish to identify : ")
            await asyncio.sleep(10)
            self._ctx.send_file(str(imagename), str("/golem/work/"+ GlobalStore.dataset + "/test/Unknown/" + imagename))
            send = yield self._ctx.commit()
            sendf = await send
            print("Test Image Sent!")
            # TODO assume test location and classes
            self._ctx.run(self.CLASSIFIERCLIENT, "-p", "/golem/work/" + GlobalStore.dataset + "/test","-c", GlobalStore.classes)
            future_results = yield self._ctx.commit()
            results = await future_results
            prediction = results[0].stdout.strip()
            #Cleanup test folder for next prediction
            self._ctx.run("/bin/rm", "-rf", "/golem/work/"+ GlobalStore.dataset +"/test")
            deletion = yield self._ctx.commit()
            ds = await deletion
            print(prediction)
        async def train(validpath, trainpath):
            datapath = input("What is the name of the training data folder : ")
            validpath = ""
            #Send in the dataset as a zipped file
            self._ctx.send_file(str(datapath), str("/golem/work/"+ GlobalStore.dataset + "/train/" + datapath))
            self._ctx.send_file(str(datapath), str("/golem/work/"+ GlobalStore.dataset + "/valid/" + validpath))
            transferred = yield self._ctx.commit()
            finished = await transferred
            train = "/golem/work/" + GlobalStore.dataset +"/train/" + datapath
            valid = "/golem/work/" + GlobalStore.dataset +"/train/" + validpath
            validpath = "/golem/work/" + GlobalStore.dataset +"/valid/" + datapath
            self._ctx.run("/bin/tar","--no-same-owner", "-C", "/golem/work/" + GlobalStore.dataset + "/train/", "-xzvf", train)
            self._ctx.run("/bin/tar","--no-same-owner", "-C", "/golem/work/" + GlobalStore.dataset + "/valid/", "-xzvf", valid)
            zipped = yield self._ctx.commit()
            finalized = await zipped
            #Now data is unzipped, next it executes
            self._ctx.run(self.CLASSIFIER, "-t", "/golem/work/" + GlobalStore.dataset + "/train", "-v", "/golem/work/" + GlobalStore.dataset + "/valid", "-c", GlobalStore.classes) 
            future_results = yield self._ctx.commit()
            results = await future_results
            print("Model Successfully Trained")
        app = Flask(__name__)
        api = Api(app)
        class PredictTestService:
            def post(self):
                parser = reqparse.RequestParser()
                parser.add_argument('task', required=True)  # "predict" or "train"
                parser.add_argument('validpath', required=False) # For Train
                parser.add_argument('trainpath', required=False) # For Train
                parser.add_argument('imagename', required=False) # For Predict
                args = parser.parse_args()
                if args.task == 'predict':
                    result = predict(args.imagename)
                elif args.task == 'train':
                    result = train(args.validpath, args.trainpath)
                # TODO call methods with this data
                return {"response" : result}, 200
        while True:
            task = ""
            app.run()
        

async def main(subnet_tag, driver=None, network=None):
    async with Golem(
        budget=4.00,
        subnet_tag=subnet_tag,
        driver=driver,
        network=network,
    ) as golem:

        print(
            f"yapapi version: {yapapi_version}\n"
            f"Using subnet: {subnet_tag}, "
            f"payment driver: {golem.driver}, "
            f"and network: {golem.network}\n"
        )

        commissioning_time = datetime.now()

        print(
            f"starting {pluralize(NUM_INSTANCES, 'instance')}"
        )

        # start the service

        cluster = await golem.run_service(
            ImageClassifierService,
            num_instances=NUM_INSTANCES,
            expiration=datetime.now(timezone.utc) + timedelta(minutes=120),
        )

        # helper functions to display / filter instances

        def instances():
            return [(s.provider_name, s.state.value) for s in cluster.instances]

        def still_running():
            return any([s for s in cluster.instances if s.is_available])

        def still_starting():
            return len(cluster.instances) < NUM_INSTANCES or any(
                [s for s in cluster.instances if s.state == ServiceState.starting]
            )

        # wait until instances are started

        while still_starting() and datetime.now() < commissioning_time + STARTING_TIMEOUT:
            print(f"instances: {instances()}")
            await asyncio.sleep(5)

        if still_starting():
            raise Exception(f"Failed to start instances before {STARTING_TIMEOUT} elapsed :( ...")

        print("All instances started :)")

        # allow the service to run for a short while
        # (and allowing its requestor-end handlers to interact with it)

        start_time = datetime.now()

        while datetime.now() < start_time + timedelta(minutes=120): # 2 hour timeout 
            print(f"instances: {instances()}")
            await asyncio.sleep(5)

        print(f"stopping instances")
        cluster.stop()

        # wait for instances to stop

        cnt = 0
        while cnt < 10 and still_running():
            print(f"instances: {instances()}")
            await asyncio.sleep(5)

    print(f"instances: {instances()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-c", "--classes", nargs='+')
    args = parser.parse_args()

    GlobalStore.dataset = args.dataset
    classes = ""
    for index in args.classes:
        classes = args.classes[index] + " "
    GlobalStore.classes = classes
    # This is only required when running on Windows with Python prior to 3.8:
    windows_event_loop_fix()
    now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    enable_default_logger(
        log_file=f"simple-service-yapapi-{now}.log",
        debug_activity_api=True,
        debug_market_api=True,
        debug_payment_api=True,
    )

    loop = asyncio.get_event_loop()
    task = loop.create_task(
        main(subnet_tag="devnet-beta.2", driver="zksync", network="rinkeby")
    )

    try:
        loop.run_until_complete(task)
    except NoPaymentAccountError as e:
        handbook_url = (
            "https://handbook.golem.network/requestor-tutorials/"
            "flash-tutorial-of-requestor-development"
        )
        print(
            f""
            f"No payment account initialized for driver `{e.required_driver}` "
            f"and network `{e.required_network}`.\n\n"
            f"See {handbook_url} on how to initialize payment accounts for a requestor node."
            f""
        )
    except KeyboardInterrupt:
        print(
            f""
            "Shutting down gracefully, please wait a short while "
            "or press Ctrl+C to exit immediately..."
            f""
        )
        task.cancel()
        try:
            loop.run_until_complete(task)
            print(
                f"Shutdown completed, thank you for waiting!"
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
