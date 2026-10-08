from unittest.mock import Mock

import pytest

from src.api_extensions.jobs import DeleteJobRequest, delete_job
from src.common.errors import BadRequestError, ForbiddenError, MissingResourceError
from src.status.get_info import UserInfo
from src.tagging.fabric_tagging.queue.model import CompleteJobRequest, CreateQueueItem, ListJobArgs, QueueItem


def _ended_job(jobstore, q, make_tag_args) -> QueueItem:
    job = jobstore.create_job(CreateQueueItem(qid=q.qid, model=make_tag_args().feature), auth=q.token)
    jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth=q.token)
    return job


def _act_as(resolver, user: str, is_tenant_admin: bool) -> None:
    resolver.get_user_info = Mock(return_value=UserInfo(user_adr=user, is_tenant_admin=is_tenant_admin, is_content_admin=False))


def _delete(jobstore, resolver, job: QueueItem, q, tenant: str | None = None) -> None:
    req = DeleteJobRequest(job_id=job.id, tenant=tenant, authorization=q.token)
    delete_job(req, js=jobstore, user_info_resolver=resolver)


def _assert_deleted(jobstore, job: QueueItem, q) -> None:
    with pytest.raises(MissingResourceError):
        jobstore.get_job(job.id)
    assert jobstore.list_jobs(ListJobArgs(qid=q.qid, include_unready=True), auth=q.token) == []


def test_cant_delete_job_that_hasnt_ended(jobstore, q, make_tag_args, fake_user_info_resolver):
    job = jobstore.create_job(CreateQueueItem(qid=q.qid, model=make_tag_args().feature), auth=q.token)
    _act_as(fake_user_info_resolver, job.user, is_tenant_admin=False)

    with pytest.raises(BadRequestError):
        _delete(jobstore, fake_user_info_resolver, job, q)


def test_owner_deletes_job(jobstore, q, make_tag_args, fake_user_info_resolver):
    job = _ended_job(jobstore, q, make_tag_args)
    _act_as(fake_user_info_resolver, job.user, is_tenant_admin=False)

    _delete(jobstore, fake_user_info_resolver, job, q)

    _assert_deleted(jobstore, job, q)


def test_owner_naming_their_tenant_deletes_job(jobstore, q, make_tag_args, fake_user_info_resolver):
    job = _ended_job(jobstore, q, make_tag_args)
    _act_as(fake_user_info_resolver, job.user, is_tenant_admin=False)

    _delete(jobstore, fake_user_info_resolver, job, q, tenant=job.tenant)

    _assert_deleted(jobstore, job, q)


def test_other_user_cant_delete_job(jobstore, q, make_tag_args, fake_user_info_resolver):
    job = _ended_job(jobstore, q, make_tag_args)
    _act_as(fake_user_info_resolver, "0xother", is_tenant_admin=False)

    with pytest.raises(ForbiddenError):
        _delete(jobstore, fake_user_info_resolver, job, q)


def test_tenant_admin_deletes_other_users_job(jobstore, q, make_tag_args, fake_user_info_resolver):
    job = _ended_job(jobstore, q, make_tag_args)
    _act_as(fake_user_info_resolver, "0xother", is_tenant_admin=True)

    _delete(jobstore, fake_user_info_resolver, job, q, tenant=job.tenant)

    _assert_deleted(jobstore, job, q)


def test_tenant_admin_of_another_tenant_cant_delete_job(jobstore, q, make_tag_args, fake_user_info_resolver):
    job = _ended_job(jobstore, q, make_tag_args)
    _act_as(fake_user_info_resolver, job.user, is_tenant_admin=True)

    with pytest.raises(ForbiddenError):
        _delete(jobstore, fake_user_info_resolver, job, q, tenant="iten_other")
