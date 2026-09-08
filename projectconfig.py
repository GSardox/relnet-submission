import os
from math import ceil


class BaseConfig():
    MONGODB_DATABASE_NAME = "relnet_jobs"

    CELERY_TASK_SERIALIZER = "pickle"
    CELERY_RESULT_SERIALIZER = "pickle"
    CELERY_TASK_ACKS_LATE = True
    CELERYD_PREFETCH_MULTIPLIER = 1
    BROKER_POOL_LIMIT = None
    BROKER_HEARTBEAT = 3600.0

class ProjectConfig(BaseConfig):
    mongo_port = os.environ.get("RN_MONGO_PORT", "27017")
    rabbit_port = os.environ.get("RN_RABBIT_PORT", "5672")
    cpu_concurrency = os.environ.get("RN_CPU_CONCURRENCY", "22")

    BACKEND_URL = f"mongodb://localhost:{mongo_port}/{BaseConfig.MONGODB_DATABASE_NAME}"

    laborer_pw = os.environ["RN_LABORER_PW"]
    CELERY_BROKER_URL = f"pyamqp://relnetlaborer:{laborer_pw}@localhost:{rabbit_port}/relnetvhost"

    NUMBER_WORKER_THREADS = {
        "relnet-worker-cpu": cpu_concurrency,
        "relnet-worker-gpu": '22',
        "relnet-manager": '2'
    }
    WORKER_MAX_TASKS_PER_CHILD = 1

    def get_number_worker_threads(self):
        hostname = os.environ["HOSTNAME"]
        hostname_short = hostname.split(".")[0]
        return self.NUMBER_WORKER_THREADS[hostname_short]

def get_project_config():
    app_settings = ProjectConfig()
    return app_settings
