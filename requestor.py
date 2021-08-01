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

import argparse
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


class ImageClassifierService(Service):
    CLASSIFIER = "/golem/run/ImageClassification.py"
    CLASSIFIERCLIENT = "/golem/run/ClassifierClient.py"
    @staticmethod
    async def get_payload():
        return await vm.repo(
            image_hash="af6cb3c0fb4321a7924ea6938a92af8e0fec2420ccd6f929a08e2ee0",
            min_mem_gib=8,
            min_storage_gib=20,
        )

    async def start(self):
        """
        Setup the volume so that it will play nice with the classifier and all the needed data
        is stored there. Send over the model, and make the needed dirs 
        """
        self._ctx.send_file(str("vgg16.h5"), str("/golem/work/vgg16.h5"))
        sent = yield self._ctx.commit()
        finished = await sent
        datapath = input("What is the name of the dataset to send over? : ")
        #Send in the dataset as a zipped file
        self._ctx.send_file(str(datapath), str("/golem/work/" + datapath))
        data = "/golem/work/" + datapath
        self._ctx.run("/bin/tar","--no-same-owner", "-C", "/golem/work/", "-xzvf", data)
        zipped = yield self._ctx.commit()
        finalized = await zipped
        # Next up set up some folders in the volume so the classifier can identify it
        self._ctx.run("/bin/ls", "/golem/work/dataset")
        status = yield self._ctx.commit()
        finalized = await status
        print(finalized)
        #Now data is unzipped, next it executes
    async def run(self):
        self._ctx.run("/bin/sh", "-c", "nohup python /golem/run/ImageClassification.py run &")
        self._ctx.run(self.CLASSIFIERCLIENT,"-t", "/golem/work/dataset/train", "-v", "/golem/work/dataset/valid", "--start","-c", "dog", "cat", "monkey", "cow")
        built = yield self._ctx.commit()
        done = await built
        print(done)
        while True:
            task = input("What task do you wish to run? [predict/train] : ")
            if task == "predict":
                    imagepath = input("What is the name of the image you wish to identify : ")
                    await asyncio.sleep(10)
                    self._ctx.send_file(str(imagepath), str("/golem/work/dataset/test/Unknown/" + imagepath))
                    send = yield self._ctx.commit()
                    sendf = await send
                    print("Test Image Sent!")
                    self._ctx.run(self.CLASSIFIERCLIENT, "-p", "/golem/work/dataset/test","-c","dog","cat","monkey","cow")
                    future_results = yield self._ctx.commit()
                    results = await future_results
                    prediction = results[0].stdout.strip()
                    print(prediction)
                    #print(classes[int(list(prediction.split(".")[1])[2])])
            elif task == "train":
                    datapath = input("What is the name of the training data folder : ")
                    #Send in the dataset as a zipped file
                    self._ctx.send_file(str(datapath), str("/golem/work/dataset/train/" + datapath))
                    data = "/golem/work/dataset/train/" + datapath
                    self._ctx.run("/bin/tar","--no-same-owner", "-C", "/golem/work/dataset/train/", "-xzvf", data)
                    zipped = yield self._ctx.commit()
                    finalized = await zipped
                    #Now data is unzipped, next it executes
                    self._ctx.run(self.CLASSIFIER, "-t", "/golem/work/dataset/train", "-v", "/golem/work/dataset/valid", "-c","dog","cat","monkey","cow") 
                    future_results = yield self._ctx.commit()
                    results = await future_results
                    #next its awaited, once this completes the model is trained
                    #but, some cleanup must be done
                    self._ctx.run("/bin/rm", "-rf", "/golem/work/dataset/train")
                    deletion = yield self._ctx.commit()
                    ds = await deletion
                    print("Model Successfully Trained")

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
    now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    parser.set_defaults(log_file=f"simple-service-yapapi-{now}.log")
    args = parser.parse_args()

    # This is only required when running on Windows with Python prior to 3.8:
    windows_event_loop_fix()

    enable_default_logger(
        log_file=args.log_file,
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
