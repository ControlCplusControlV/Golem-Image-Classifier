#!/usr/bin/env python3
"""
Below defines the requestor agent which then opens up a CLI interface
to interact with the Image Classifier Service
"""
import asyncio
from datetime import datetime, timedelta, timezone
import pathlib
import random
import string
import sys
from pathlib import Path
import time
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


class ImageClassifier(Service):
    imageClassifier = "/golem/run/imageclassifier.py"

    @staticmethod
    async def get_payload():
        return await vm.repo(
            image_hash="baa76c7c825808b8b2b14c7e091ec8be385e05a5781fabd04f2a87ee",
            min_mem_gib=5,
            min_storage_gib=7,
        )

    async def start(self):
        '''
        Runs an prediction test upon startup to verify everything is working,
        this step is optional but was helpful when debugging and decided to leave
        it in for those who choose to modify this, as it verifies prediction works
        '''
        self._ctx.run(self.imageClassifier, "--predict", "/golem/work/test2.jpg")
        yield self._ctx.commit()

    async def run(self):
        print("Service Successfully Initialized!")
        while True:
            '''
            Uses the user input function, and http webserver could be done for this but
            as a lot of AI work is done on "deskside compute" I thought a CLI would be easier
            for Data Scientists rather than a webserver interface
            '''
            task = input("What task do you wish to run? [predict/train] : ")
            if task == "predict":
                    imagepath = input("What is the name of the image you wish to identify : ")
                    await asyncio.sleep(10)
                    self._ctx.send_file(str(imagepath), str(f"/golem/work/{imagepath}"))
                    print("Test Image Sent!")
                    testpath = "/golem/work/" + imagepath
                    self._ctx.run(self.imageClassifier, "--predict", testpath)  
                    future_results = yield self._ctx.commit()
                    results = await future_results
                    print(results)
            elif task == "train":
                    datapath = input("What is the name of the h5 data file :")
                    labelpath = input("What is the name of the h5 labels file :")
                    datapaths = "/golem/work/" + datapath
                    labelpaths = "/golem/work/" + labelpath
                    self._ctx.send_file(str(datapath), str(datapaths))
                    self._ctx.send_file(str(labelpath), str(labelpaths))
                    self._ctx.run(self.imageClassifier, "--traindata", datapaths, "--trainlabels", labelpaths) 
                    print("Model Successfully Trained")

async def main(subnet_tag, driver=None, network=None):
    async with Golem(
        budget=1.0,
        subnet_tag=subnet_tag,
        driver=driver,
        network=network,
    ) as golem:
        '''
        Most of the helper code from the Simple Service Model is kept in as the information provided
        is very helpful, 
        '''
        print(
            f"yapapi version: {yapapi_version}\n"
            f"Using subnet: {subnet_tag}, "
            f"payment driver: {golem.driver}, "
            f"and network: {golem.network}\n"
        )

        commissioning_time = datetime.now()

        print(f"Starting the Image Classifier Service")

        # start the service

        cluster = await golem.run_service(
            ImageClassifier,
            num_instances=1,
            expiration=datetime.now(timezone.utc) + timedelta(minutes=120),
        )

        # helper functions to display / filter instances

        def instances():
            return [(s.provider_name, s.state.value) for s in cluster.instances]

        def still_running():
            return any([s for s in cluster.instances if s.is_available])

        def still_starting():
            return len(cluster.instances) < 1 or any(
                [s for s in cluster.instances if s.state == ServiceState.starting]
            )
        while still_starting() and datetime.now() < commissioning_time + STARTING_TIMEOUT:
            print(f"instances: {instances()}")
            await asyncio.sleep(5)

        if still_starting():
            raise Exception(f"Failed to start instances before {STARTING_TIMEOUT} elapsed :( ...")
        start_time = datetime.now()

        while datetime.now() < start_time + timedelta(minutes=2):
            print(f"instances: {instances()}")
            await asyncio.sleep(15)

        print(f"stopping instances")
        cluster.stop()

        cnt = 0
        while cnt < 10 and still_running():
            print(f"instances: {instances()}")
            await asyncio.sleep(5)

    print(f"instances: {instances()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    parser.set_defaults(log_file=f"imageclassifier-service-yapapi-{now}.log")
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
        '''
        Error handling also kept in as new people using this who may have improperly
        setup their requestor node can benefit from it
        '''
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
