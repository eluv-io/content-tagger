from dataclasses import asdict
from unittest.mock import patch

def test_tenant_status(mock_app, make_tag_args):
    client = mock_app.test_client()
    # tag first
    args = make_tag_args(feature="test_feature")
    qid = "qid1"

    response = client.post(
        f"/{qid}/tag?authorization=fake", 
        json={
            "options": {
                "destination_qid": "",
                "replace": True,
                "max_fetch_retries": 3,
                "scope": {}
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["hello1", "hello2"]},
                }
            ]
        }
    )

    assert response.status_code == 200
    response = client.get("/job-status?tenant=test%20tenant&authorization=fake")
    assert response.status_code == 200
    jobs = response.get_json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["tenant"] == "test tenant"