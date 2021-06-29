import asyncio
from datetime import datetime, timedelta

from yapapi import Golem
from yapapi.services import Service
from yapapi.log import enable_default_logger
from yapapi.payload import vm

REFRESH_INTERVAL_SEC = 5


class ClassifierService(Service):
	CLASSIFIER = "imageclassifier.py"
	MODEL = "classifier.pkl"
	@staticmethod
	async def get_payload():
		return await vm.repo(
			image_hash="f2c8070ab46988c432c4d70e19e52e0cb008647b83dc5b77900516c59ba7d437",
			min_mem_gib=4,
			min_storage_gib=7.0,
		)

	async def start(self):
		# handler responsible for starting the service
		self._ctx.run(self.CLASSIFIER, "--trainmodel", "True")
		outcome = yield self._ctx.commit()
		if outcome == True:
			print("Model Successfully Trained")

	async def run(self):
		while True:
			await asyncio.sleep(10)
			self._ctx.run(self.CLASSIFIER, "--predict", "True")  # idx 1

			prediction = yield self._ctx.commit()
			results = await prediction

			print(results)
async def main():
	async with Golem(budget=1.0, subnet_tag="devnet-beta.2") as golem:
		cluster = await golem.run_service(ClassifierService, num_instances=1)


		while datetime.now() < start_time + timedelta(minutes=1):
			for num, instance in enumerate(cluster.instances):
				print(f"Instance {num} is {instance.state.value} on {instance.provider_name}")
				await asyncio.sleep(REFRESH_INTERVAL_SEC)
if __name__ == "__main__":
	enable_default_logger(log_file="hello.log")
	loop = asyncio.get_event_loop()
	task = loop.create_task(main())
