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
import socket
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
    CLASSIFIER = "/golem/run/imageclassifier.py"

    @staticmethod
    async def get_payload():
        return await vm.repo(
            image_hash="262c82b1769f062b181095b15d0591f313a1ce9748c8fb53fc7df500",
            min_mem_gib=8,
            min_storage_gib=20,
        )

    async def start(self):
        """
        Setup the volume so that it will play nice with the classifier and all the needed data
        is stored there. Send over the model, and make the needed dirs
        """

        status = self._ctx.send_file(
            "model.tar.gz", "/golem/work/model.tar.gz")
        status = yield self._ctx.commit()
        transfer = await status
        print("Model Transferred")
        self._ctx.run(
            "/bin/tar",
            "--no-same-owner",
            "-C",
            "/golem/work/",
            "-xzvf",
            "/golem/work/model.tar.gz")
        zipped = yield self._ctx.commit()
        finalized = await zipped
        # Next up set up some folders in the volume so the classifier can
        # identify it
        self._ctx.run("/bin/mkdir", "/golem/work/dataset")
        dataset = yield self._ctx.commit()
        datasetf = await dataset
        self._ctx.run("/bin/mkdir", "/golem/work/dataset/test")
        test = yield self._ctx.commit()
        testf = await test
        self._ctx.run("/bin/mkdir", "/golem/work/dataset/test/Unknown")
        utest = yield self._ctx.commit()
        utestf = await utest
        self._ctx.run("/bin/ls", "/golem/work/dataset/test")
        files = yield self._ctx.commit()
        rfiles = await files
        print(rfiles)

    async def run(self):
        while True:
            task = input("What task do you wish to run? [predict/train] : ")
            if task == "predict":
                imagepath = input(
                    "What is the name of the image you wish to identify : ")
                await asyncio.sleep(10)
                self._ctx.send_file(
                    str(imagepath), str(
                        "/golem/work/dataset/test/Unknown/" + imagepath))
                send = yield self._ctx.commit()
                sendf = await send
                print("Test Image Sent!")
                self._ctx.run(
                    self.CLASSIFIER,
                    "--predict",
                    "/golem/work/dataset/test",
                    "--batch",
                    "1")
                future_results = yield self._ctx.commit()
                results = await future_results
                classes = ["bluebell", "buttercup", "coltsfoot", "cowslip"]
                prediction = results[0].stdout.strip()
                print(classes[int(list(prediction.split(".")[1])[2])])
            elif task == "train":
                datapath = input(
                    "What is the name of the training data folder : ")
                # Send in the dataset as a zipped file
                self._ctx.send_file(
                    str(datapath), str(
                        "/golem/work/dataset/train/" + datapath))
                data = "/golem/work/dataset/train/" + datapath
                self._ctx.run(
                    "/bin/tar",
                    "--no-same-owner",
                    "-C",
                    "/golem/work/dataset/train/",
                    "-xzvf",
                    data)
                zipped = yield self._ctx.commit()
                finalized = await zipped
                # Now data is unzipped, next it executes
                self._ctx.run(
                    self.CLASSIFIER,
                    "--trainloc",
                    "/golem/work/dataset/train")
                future_results = yield self._ctx.commit()
                results = await future_results
                # next its awaited, once this completes the model is trained
                # but, some cleanup must be done
                self._ctx.run("/bin/rm", "-rf", "/golem/work/dataset/train")
                deletion = yield self._ctx.commit()
                ds = await deletion
                # Model Cleared, so going to re-create the folder for future training
                #self._ctx.run("/bin/mkdir", "/golem/work/dataset/test")
                # test = yield self._ctx.commit()
                # testf = await test
                print("Model Successfully Trained")


async def main(subnet_tag, driver=None, network=None):
    async with Golem(
        budget=5.00,
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
            return [(s.provider_name, s.state.value)
                    for s in cluster.instances]

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
            raise Exception(
                f"Failed to start instances before {STARTING_TIMEOUT} elapsed :( ...")

        print("All instances started :)")

        # allow the service to run for a short while
        # (and allowing its requestor-end handlers to interact with it)

        start_time = datetime.now()

        while datetime.now() < start_time + timedelta(minutes=120):  # 2 hour timeout
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
            f"")
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
