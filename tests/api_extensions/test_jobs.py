from src.api_extensions.jobs import delete_job
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, ListJobArgs


def test_delete_job(jobstore, make_tag_args):
    args = make_tag_args()
    job = jobstore.create_job(CreateQueueItem(qid="iq__test", params=args, status_details=None, additional_info={}), auth="token")

    delete_job(job.id, auth="token", js=jobstore)

    remaining = jobstore.list_jobs(args=ListJobArgs(), auth="token")
    assert all(j.id != job.id for j in remaining)
