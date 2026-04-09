from unittest.mock import Mock

import pytest

from src.api_extensions.jobs import DeleteJobRequest, delete_job
from src.common.errors import BadRequestError, ForbiddenError, MissingResourceError
from src.status.get_info import UserInfo
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, ListJobArgs, UpdateJobRequest


def test_delete_job(jobstore, make_tag_args, fake_user_info_resolver):
    args = make_tag_args()
    job = jobstore.create_job(CreateQueueItem(qid="iq__test", params=args, status_details=None, additional_info={}), auth="token")
    req = DeleteJobRequest(job_id=job.id, tenant=None, authorization="token")

    with pytest.raises(BadRequestError):
        # can't delete a job that isn't in a terminal state
        delete_job(req, js=jobstore, user_info_resolver=fake_user_info_resolver)

    # update job state
    jobstore.update_job(UpdateJobRequest(id=job.id, status="succeeded"), "token")

    delete_job(req, js=jobstore, user_info_resolver=fake_user_info_resolver)

    assert jobstore.get_job(job.id).status == "deleted"

    job = jobstore.create_job(CreateQueueItem(qid="iq__test", params=args, status_details=None, additional_info={}), auth="token")

    # update job state
    jobstore.update_job(UpdateJobRequest(id=job.id, status="failed"), "token")

    fake_user_info_resolver.get_user_info = Mock(return_value=UserInfo(
        user_adr="0x456",
        is_tenant_admin=False,
        is_content_admin=False
    ))

    req = DeleteJobRequest(job_id=job.id, tenant=None, authorization="token")

    with pytest.raises(ForbiddenError):
        delete_job(req, js=jobstore, user_info_resolver=fake_user_info_resolver)

    # ok if tenant_admin
    fake_user_info_resolver.get_user_info = Mock(return_value=UserInfo(
        user_adr="0x456",
        is_tenant_admin=True,
        is_content_admin=False
    ))
        
    tenant_id = fake_user_info_resolver.get_tenant(qid="iq__test", token="token")

    req = DeleteJobRequest(job_id=job.id, tenant=tenant_id, authorization="token")

    delete_job(req, js=jobstore, user_info_resolver=fake_user_info_resolver)

    assert jobstore.get_job(job.id).status == "deleted"

    # check that no deleted jobs come up in list
    jobs = jobstore.list_jobs(ListJobArgs(qid="iq__test"), auth="token")
    for j in jobs:
        assert j.status != "deleted"

    # check that if user specifies tenant and is not tenant admin, it will still be ok as long as the job belongs to them
    job = jobstore.create_job(CreateQueueItem(qid="iq__test", params=args, status_details=None, additional_info={}), auth="token")
    jobstore.update_job(UpdateJobRequest(id=job.id, status="succeeded"), "token")

    fake_user_info_resolver.get_user_info = Mock(return_value=UserInfo(
        user_adr="0x456",
        is_tenant_admin=False,
        is_content_admin=False
    ))

    req = DeleteJobRequest(job_id=job.id, tenant="tenant1", authorization="token")

    delete_job(req, js=jobstore, user_info_resolver=fake_user_info_resolver)

    assert jobstore.get_job(job.id).status == "deleted"

    # check that mismatched tenant id raises ForbiddenError
    job = jobstore.create_job(CreateQueueItem(qid="iq__test", params=args, status_details=None, additional_info={}), auth="token")
    jobstore.update_job(UpdateJobRequest(id=job.id, status="succeeded"), "token")

    fake_user_info_resolver.get_user_info = Mock(return_value=UserInfo(
        user_adr="0x123",
        is_tenant_admin=True,
        is_content_admin=False
    ))

    req = DeleteJobRequest(job_id=job.id, tenant="some_tenant_id", authorization="token")

    with pytest.raises(ForbiddenError):
        delete_job(req, js=jobstore, user_info_resolver=fake_user_info_resolver)