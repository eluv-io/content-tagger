# Tagging flow

## Start tagging

```bash
curl -X POST "http://localhost:8086/iq__b7ZBuXBYAqiwCc5oirFZEdfWY6v/tag?authorization=$AUTH_TOKEN" -d '{"features":{"asr":{}, "shot":{}, "caption":{}}}' -H "Content-Type: application/json"
```

#### Expected response 

```json
{"message": "Tagging started on iq__b7ZBuXBYAqiwCc5oirFZEdfWY6v"}
```

## Check status

```bash
curl -X GET "http://localhost:8086/iq__b7ZBuXBYAqiwCc5oirFZEdfWY6v/status?authorization=$AUTH_TOKEN"
```
#### Example responses

Phase of tagging is given in the "status". Once the actual tagging begins, the "tagging progress" will be shown as a ratio indicating the number of parts tagged. 

Pre-tagging

```
{
  "audio": {
    "asr": {
      "status": "Fetching content",
      "time_running": 2.6280276775360107,
      "tagging_progress": "",
      "tag_job_id": null,
      "error": null,
      "failed": []
    }
  },
  "video": {
    "shot": {
      "status": "Fetching content",
      "time_running": 2.626570701599121,
      "tagging_progress": "",
      "tag_job_id": null,
      "error": null,
      "failed": []
    },
    "caption": {
      "status": "Starting",
      "time_running": 2.625352144241333,
      "tagging_progress": "",
      "tag_job_id": null,
      "error": null,
      "failed": []
    }
  }
}
```

During tagging

```
{
  "audio": {
    "asr": {
      "status": "Tagging content",
      "time_running": 6.0835206508636475,
      "tagging_progress": "0/8",
      "tag_job_id": "f9a9bfde-0dda-4454-b49a-cf1bda24e446",
      "error": null,
      "failed": []
    }
  },
  "video": {
    "shot": {
      "status": "Tagging content",
      "time_running": 6.082131385803223,
      "tagging_progress": "0/8",
      "tag_job_id": "c84d590e-658f-4667-88e4-8b0e9dd35489",
      "error": null,
      "failed": []
    },
    "caption": {
      "status": "Tagging content",
      "time_running": 6.080943584442139,
      "tagging_progress": "0/8",
      "tag_job_id": "256266e8-8c64-4883-99c0-0d38720b3322",
      "error": null,
      "failed": []
    }
  }
}
```

Finished tagging

```
{
  "audio": {
    "asr": {
      "status": "Completed",
      "time_running": 55.373307943344116,
      "tagging_progress": "8/8",
      "tag_job_id": "f9a9bfde-0dda-4454-b49a-cf1bda24e446",
      "error": null,
      "failed": []
    }
  },
  "video": {
    "shot": {
      "status": "Completed",
      "time_running": 20.33554458618164,
      "tagging_progress": "8/8",
      "tag_job_id": "c84d590e-658f-4667-88e4-8b0e9dd35489",
      "error": null,
      "failed": []
    },
    "caption": {
      "status": "Completed",
      "time_running": 435.7549319267273,
      "tagging_progress": "8/8",
      "tag_job_id": "256266e8-8c64-4883-99c0-0d38720b3322",
      "error": null,
      "failed": []
    }
  }
}
```

## Finalize

Right now this will only upload tags if one of the tracks is fully finished. Else it will give error. 

```bash
curl -X POST "http://localhost:8086/iq__3VtpCUVv4DRdjAiPgoDYuJZb4Abp/finalize?write_token=$WRITE_TOKEN&authorization=$AUTH_TOKEN&force=true" | jq
```

#### Expected response

```json
{
  "message": "Succesfully uploaded tag files. Please finalize the write token.",
  "write token": "tqw__HSX6TnUJpn2hk7yQZj6ho2fYaJxF3DJsbrA9HT5ufvWvBJtMbShr89bXppWicWu7PT2RVP7w2dHfafJUrP8"
}
```