#!/usr/bin/env node

const readline = require('readline');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const yargs = require('yargs');
const { hideBin } = require('yargs/helpers');

// Configuration
const server = process.env.TAGGERV2_URL || "http://localhost:8086";
const tagstore = process.env.TAGSTORE_URL || "https://ai.contentfabric.io/tagstore";

// Global state to mimic Python script
let written = {};

// Default configs
const assets_params = {
    "options": {
        "destination_qid": "",
        "replace": true,
        "max_fetch_retries": 3,
        "scope": {"type": "assets"}
    },
    "jobs": [
        {"model": "logo", "model_params": {}},
        {"model": "ocr", "model_params": {}},
        {"model": "llava", "model_params": {"model": "elv-llamavision:1"}},
        {"model": "caption", "model_params": {}},
        {"model": "asr", "model_params": {}},
        {"model": "shot", "model_params": {}}
    ]
};

const video_params = {
    "jobs": [
        {"model": "caption", "model_params": {"fps": 0.33}}
    ]
};

//////////////////////////////////////////
// --- Helper Functions and Classes --- // 
//////////////////////////////////////////
 class PromiseWorker {
  
  #promiseSlots = []  
  #work = []
  #workAvailable = null
  #resolveWork = null
  #servicing = true
  
  constructor(slots = 3) {    
    this.#promiseSlots = new Array(slots).fill(null)
    this.#workAvailable = new Promise((resolve) => { this.#resolveWork = resolve })
    this.log = function() {}
  }

  addWork(job) {
    const ret = new Promise(async (resolve, reject) => {
      this.#work.push({ job, resolve, reject} )
    })
    
    if (this.#work.length == 1) {
      this.#resolveWork()
      this.#workAvailable = new Promise((resolve) => { this.#resolveWork = resolve })
    }
    
    return ret
  }

  async servicer() {

    this.log("performwork " + this.#work.length)
    
    while (true) {
      let workelement = undefined

      while (workelement === undefined && this.#servicing) {
        workelement = this.#work.shift()
        if (workelement === undefined) await this.#workAvailable        
      }

      if (this.#servicing == false) break
      
      this.log("exec work", workelement)

      const job = workelement.job
      const callerResolve = workelement.resolve
      const callerReject = workelement.reject

      let slot
      while (true) {        
        slot = this.#promiseSlots.findIndex( (v) => v == null)            
        
        if (slot >= 0) break
        
        if (!this.#promiseSlots.some( (v) => v != null )) {
          console.error("No free slots but no in use slots either")
          throw new Error("No free slots but no in use slots either")
        }
        
        const [done] = await Promise.race(this.#promiseSlots.filter( (v) => v != null))
        this.log("finished slot", done)
        this.#promiseSlots[done] = null
      }

      if (this.#servicing == false) break
      
      this.#promiseSlots[slot] = new Promise( async (resolve, reject) => {
        try { 
          this.log(`(${slot}) WORKER START ${slot}`)
          const result = await job()
          this.log(`(${slot}) WORKER DONE ${slot} RESULT ${result}`)
          resolve([slot])
          callerResolve(result)
          return
        }
        catch (err) {
          this.log(`(${slot}) WORKER DONE ${slot} ERR ${err}`)              
          resolve([slot])
          callerReject(err)
          return
        }
      })      
    }
    this.#work = null
    this.#promiseSlots = null    
  }
  
  stopService() {
    this.#servicing = false
    if (this.#work.length == 0) {
      this.#resolveWork()
      this.#workAvailable = new Promise((resolve) => { this.#resolveWork = resolve })
    }
  } 
}

class ReadlineInput {
  #currentTimeout = 0
  #rl = null
  #currentResolver = null
  #currentAbortController = new AbortController()
  #currentTimeoutId = null
  #history = []
  #nonTTYiterator = null

  constructor() {
    this._ondata    = this.#ondata.bind(this)
    this._onclose   = this.#onclose.bind(this)
    this._onhistory = this.#onhistory.bind(this)
    this._ontimeout = this.#ontimeout.bind(this)
    this._callback  = this.#callback.bind(this)
  }

  #ondata(d) {
    this.#setTimeout()     // reset timeout any time we get any stdin data
  }

  #onclose() {
    this.#callback(null)   // close of readline means return null to awaiting promise
    process.stdout.write("\n")
  }

  #onhistory(histup) {
    this.#history = histup // history updated, keep track of it
  }

  #ontimeout() {
    // timeout fired, abort the input then return 0 to awaiting promise
    this.#currentAbortController.abort()
    this.#currentAbortController = new AbortController()
    this.#currentTimeoutId = null
    this.#callback(0)
  }

  #setTimeout() {
    if (this.#currentTimeout > 0) {
      if (this.#currentTimeoutId != null) clearTimeout(this.#currentTimeoutId)
      this.#currentTimeoutId = setTimeout(this._ontimeout, this.#currentTimeout)
    }
  }

  // (internal) callback passed to readline question, and used by other events to return the value thru the promise
  #callback(data) {
    process.stdin.removeListener('data', this._ondata)
    if (this.#currentTimeoutId != null) clearTimeout(this.#currentTimeoutId)
    this.#currentTimeoutId = null

    if (this.#currentResolver) {
      this.#rl.removeListener('close',   this._onclose)
      this.#rl.removeListener('history', this._onhistory)
      this.#rl.removeListener('SIGCONT', this.#rl.resume)

      this.#rl.close()
      this.#rl = null

      this.#currentResolver(data)
      this.#currentResolver = null
    }
    else {
      console.error("STALE RESOLVE?? " + data)
    }
  }

  async question(query, timeout = 0) {
    if (this.#currentResolver) return Promise.reject("Previous question did not settle yet.")
    if (timeout != null && timeout > 0 && !process.stdin.isTTY) return Promise.reject("timeout used on non-tty input")

    if (!this.#rl) {
      this.#rl = readline.createInterface({
        input: process.stdin,
        output: process.stdin.isTTY ? process.stdout : null,
        terminal: process.stdin.isTTY,
        history: this.#history
      });

      if (process.stdin.isTTY) {
        this.#rl.on('history', this._onhistory)
        this.#rl.on('SIGCONT', this.#rl.resume)
      }
    }

    if (!process.stdin.isTTY) {
      if (!this.#nonTTYiterator) this.#nonTTYiterator = await this.#rl[Symbol.asyncIterator]()
      const line = (await this.#nonTTYiterator.next()).value
      return (line === undefined) ? null : line
    }

    const promise = new Promise((resolve) => {
      this.#currentResolver = resolve
      this.#currentAbortController = new AbortController()
      this.#currentTimeout = timeout
    })

    this.#rl.on('close', this._onclose)

    if (timeout > 0) {
      process.stdin.on('data', this._ondata)
      this.#setTimeout()
    }

    this.#rl.question(query, { signal: this.#currentAbortController.signal }, this._callback)

    return promise
  }
}

function formatTime(t) {
  t = parseInt(t);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t / 60) % 60);
  const s = t % 60;
  return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
};

// fetch handle errors
async function fetch_dict_with_status(...args) {
  let resp
  try {
    resp = await fetch(...args)
  }
  catch (e) {
    return {
      "error": "could not do fetch: " + e,
      "cause": e.cause,
      "status": -1,
      "content": "no content"
    }
  }

  let text
  try {
    text = await resp.text()
  } catch (e) {
    return {
      "error": "could not read response body: " + e,
      "status": resp.status
    }
  }

  try {
    const res = JSON.parse(text)
    if (Array.isArray(res) || typeof res != "object") {
      return {
        "status": resp.status,
        "content": res
      }
    }
    else {
      res.status = resp.status
      return res
    }
  } catch (e) {
    return {
      "error": "could not parse json: " + e,
      "status": resp.status,
      "content": text
    }
  }
}

function get_auth(config, qhit) {
    const cmd = `qfab_cli content token create ${qhit} --update --config ${config}`;
    try {
        const out = execSync(cmd).toString();
        const token = JSON.parse(out).bearer;
        return token;
    } catch (e) {
        console.error("Error getting auth token:", e.message);
        throw e;
    }
}

function get_write_token(qhit, config) {
    if (qhit.startsWith("tqw")) {
        return qhit;
    }
    const cmd = `qfab_cli content edit ${qhit} --config ${config}`;
    try {
        const out = execSync(cmd).toString();
        const write_token = JSON.parse(out).q.write_token;
        return write_token;
    } catch (e) {
        console.error("Error getting write token:", e.message);
        throw e;
    }
}

async function get_status(qhit, auth) {
    const url = new URL(`${server}/${qhit}/status`);
    url.searchParams.append("authorization", auth);

    const response_data = await fetch_dict_with_status(url);

    if (response_data && response_data.jobs) {
        const reports = response_data.jobs;
        const status = {};
        for (const report of reports) {
            const stream = report.stream;
            const model = report.model;
            if (!status[stream]) {
                status[stream] = {};
            }
            status[stream][model] = {
                'status': report.status,
                'tagging_progress': report.tagging_progress,
                'time_running': report.time_running,
                'failed': report.failed,
                'missing_tags': report.missing_tags
            };
            if (report.message) {
                status[stream][model].message = report.message;
            }
        }
        return status;
    } else if (response_data && response_data.error) {
        return response_data;
    } else {
        return response_data;
    }
}

async function tag(contents, auth, assets, params, startTime = null, endTime = null) {

    for (let i = 0; i < contents.length; i++) {
        const qhit = contents[i];
        let url;
        if (assets) {
            url = `${server}/${qhit}/image_tag`;
        } else {
            url = `${server}/${qhit}/tag`;
        }

        // Deep copy params to avoid modifying the original for subsequent iterations
        const currentParams = JSON.parse(JSON.stringify(params));

        // Update scope
        if (startTime !== null || endTime !== null) {
            if (!currentParams.options) currentParams.options = {};
            if (!currentParams.options.scope) currentParams.options.scope = {};

            if (startTime !== null) currentParams.options.scope.start_time = parseInt(startTime);
            if (endTime !== null) currentParams.options.scope.end_time = parseInt(endTime);
        }

        console.log(JSON.stringify(currentParams, null, 2));

        const urlObj = new URL(url);
        urlObj.searchParams.append("authorization", auth);

        const res = await fetch_dict_with_status(urlObj, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentParams)
        });
        console.log(res)

        const sleepTime = parseFloat(process.env.TAGGERV2_START_SLEEP || 0);
        if (sleepTime > 0) {
            await new Promise(resolve => setTimeout(resolve, sleepTime * 1000));
        }
    }
}

function commit(write_token, config) {
    const cmd = `qfab_cli content finalize ${write_token} --config ${config}`;
    try {
        const out = execSync(cmd).toString();
        console.log(out);
    } catch (e) {
        console.error("Error committing:", e.message);
    }
}

async function write(qhit, config, do_commit, force = false, leave_open = false) {
    const auth_token = get_auth(config, qhit);
    const write_token = get_write_token(qhit, config);
    const write_url = new URL(`${tagstore}/${qhit}/write`);
    write_url.searchParams.append("write_token", write_token);

    const respdict = await fetch_dict_with_status(write_url, {
        method: 'POST',
        headers: { "Authorization": `Bearer ${auth_token}` }
    });
    console.log(respdict);

    if (do_commit && respdict.status === 200) {
        commit(write_token, config);
    }
    return write_token;
}

async function aggregate(qhit, config, do_commit) {
    console.log("");
    console.log("****************************");
    console.log("Aggregating is not necessary (tagstore does it on write)");
    console.log("");
    console.log("Pausing 5 seconds so you can ctrl-c if you want to.");
    console.log("****************************");
    console.log("");

    if (process.env.TAGGERV3_AGG_NODELAY === undefined) {
        await new Promise(resolve => setTimeout(resolve, 5000));
    }

    const auth_token = get_auth(config, qhit);
    const write_token = get_write_token(qhit, config);
    const aggregate_url = new URL(`https://ai.contentfabric.io/tagging/${qhit}/aggregate`);
    aggregate_url.searchParams.append("authorization", auth_token);
    aggregate_url.searchParams.append("write_token", write_token);
    aggregate_url.searchParams.append("replace", "true");

    const respdict = await fetch_dict_with_status(aggregate_url, { method: 'POST' });
    console.log(respdict);

    if (do_commit && !respdict.error) {
        commit(write_token, config);
    }
    return write_token;
}

async function write_all(contents, config, do_commit, force = false) {
    for (const qhit of contents) {
        if (written[qhit]) {
            console.log(`${qhit} already written, clearwritten to clear list`);
            continue;
        }

        console.log(`Finalizing ${qhit} force = ${force}`);
        try {
            let leave_open = false;
            let current_do_commit = do_commit;
            if (qhit.startsWith("tqw")) {
                leave_open = true;
                current_do_commit = false;
            }

            await write(qhit, config, current_do_commit, force, leave_open);
            written[qhit] = true;
        } catch (e) {
            console.log(`${e} while finalizing ${qhit}`);
        }
    }
}

async function stop(qhit, auth, models) {
    const params = new URLSearchParams({ authorization: auth });
    for (const model of models) {
        const url = `${server}/${qhit}/stop/${model}`;
        try {
            const urlObj = new URL(url);
            urlObj.search = params.toString();
            const res = await fetch_dict_with_status(urlObj, { method: 'POST' });
            if (res.status === 200) {
                console.log(`Successfully stopped tagging for ${qhit} on model ${model}.`);
            } else {
                console.log(`Failed to stop tagging for ${qhit} on model ${model}: ${res.status} ${res.statusText}`);
            }
        } catch (e) {
            console.log(`Error while stopping tagging for ${qhit} on model ${model}: ${e}`);
        }
    }
}

async function list_models() {
    console.log("getting model list:");
    const modresp = await fetch_dict_with_status(`${server}/list`);
    return modresp.content;
}

function help() {
    console.log(`
t,tag [iq_regex] [model]        tag content
                                if iq_regex specified, only tag matching
                                if model specified, only start that model
stop [iq] [model]               stop tagging
                                if iq given, only stop for that iq (must be full iq, not regex)
                                if model given, only stop for that model
s,status                        show status
qs [regex]                      quick status, if regex given, match output only containing regex
list                            list models tagger knows about
cw,clearwritten                 clear the "written" state to allow re-writing
h,help                          this help`);
}

function get_available_models(tag_config) {
    if (tag_config.jobs) {
        return tag_config.jobs.map(job => job.model);
    }
    return [];
}

async function quick_status(auth, qhit, filter = null) {
    if (filter === "") filter = null;
    const url = new URL(`${server}/${qhit}/status`);
    url.searchParams.append("authorization", auth);

    const status_data = await fetch_dict_with_status(url);
    if (status_data.status != 200 && !status_data.error) status_data.error = `http ${status_data.status}`
  
    if (status_data && (status_data.error || status_data.status != 200)) {
        const line = `[${"".padStart(9)}] ${qhit.padEnd(32)} / err: ${status_data.error}`;
        if (filter === null || (new RegExp(filter)).test(line)) {
            console.log(line);
        }
        return;
    }

    if (status_data && status_data.jobs) {
        const reports = status_data.jobs;
        for (const report of reports) {
            const model = report.model;
            const stream = report.stream;
            const progress = report.tagging_progress || '';
            const status = report.status || '??';
            const line = `[${String(progress).padStart(9)}] ${qhit.padEnd(32)} / (${stream}) ${model}: ${status}`;
            if (filter === null || (new RegExp(filter)).test(line)) {
                console.log(line);
            }
        }
        return;
    }

    // Handle old dict format
    if (status_data && typeof status_data === 'object') {
        for (const [imgorvid, models] of Object.entries(status_data)) {
            if (imgorvid === "error") continue;
            for (const [model, stat] of Object.entries(models)) {
                const progress = stat.tagging_progress || "";
                const status = stat.status || "??";
                const line = `[${String(progress).padStart(9)}] ${qhit.padEnd(32)} / (${imgorvid}) ${model}: ${status}`;
                if (filter === null || (new RegExp(filter)).test(line)) {
                    console.log(line);
                }
            }
        }
    }
}

// --- Main Execution ---

async function main() {
    const argv = yargs(hideBin(process.argv))
        .option('contents', {
            alias: 'c',
            type: 'string',
            description: 'filename with list of contents (iq\'s) to tag'
        })
        .option('iq', {
            alias: 'q',
            type: 'array',
            default: [],
            description: 'content (iq) to tag (specified directly)'
        })
        .option('assets', {
            type: 'boolean',
            description: 'if set, tag assets instead of videos'
        })
        .option('config', {
            type: 'string',
            description: 'fabric config file to use for making tokens'
        })
        .option('tag-config', {
            type: 'string',
            default: '',
            description: 'Tagger config json. Use @ to read a file'
        })
        .option('commit', {
            alias: 'finalize',
            type: 'boolean',
            description: 'if set, commit (finalize) on fabric after writing on tagger'
        })
        .option('start-time', {
            type: 'number',
            default: 0,
            description: 'start time in seconds'
        })
        .option('end-time', {
            type: 'number',
            default: null,
            description: 'end time in seconds'
        })
        .option('replace', {
            type: 'boolean',
            default: false,
            description: 'force replaces'
        })
        .check((argv) => {
            if (!argv.contents && (!argv.iq || argv.iq.length === 0)) {
                throw new Error("One of arguments -c/--contents or -q/--iq is required");
            }
            return true;
        })
        .parse();

    let tag_config;
    if (argv['tag-config'] !== "") {
        tag_config = argv['tag-config'];
        if (tag_config.startsWith('@')) {
            const confFile = tag_config.substring(1);
            console.log("reading tag config...");
            tag_config = JSON.parse(fs.readFileSync(confFile, 'utf8'));
        } else {
            try {
                tag_config = JSON.parse(tag_config);
            } catch (e) {}
        }
    } else {
        if (argv.assets) {
            tag_config = JSON.parse(JSON.stringify(assets_params));
        } else {
            tag_config = JSON.parse(JSON.stringify(video_params));
        }
    }

    if (argv.replace) {
        if (!tag_config.defaults) tag_config.defaults = {};
        tag_config.defaults.replace = true;
    }

    let contents = [];
    if (argv.contents) {
        console.log("reading contents...");
        const fileContent = fs.readFileSync(argv.contents, 'utf8');
        contents = fileContent.split('\n').map(line => line.trim()).filter(line => line.length > 0);
        if (contents.length === 0) {
            throw new Error("No contents found in file.");
        }
    }

    if (argv.iq && argv.iq.length > 0) {
        contents = contents.concat(argv.iq);
    }

    console.log("getting auth...");
    const auth = get_auth(argv.config, contents[0]);

    let start_time = parseInt(argv['start-time']);
    let end_time = argv['end-time'] !== null ? parseInt(argv['end-time']) : null;

    let quickstatus_watch = null;
    let models = get_available_models(tag_config);

    const rl = new ReadlineInput()

    if (process.stdin.isTTY) help()

    while (true) {
        let user_line = "";
        let timeout = null;
        if (quickstatus_watch != null) timeout = 60 * 1000;

        let answer = await rl.question(`${server} > `, timeout);

        if (answer === 0) {
          // 0 means timeout
          if (quickstatus_watch) {
            user_line = quickstatus_watch;
          } else {
            continue;
          }
        } else {
          user_line = answer;
        }

        if (user_line === null) break

        if (quickstatus_watch && user_line === quickstatus_watch) {
             console.log("[auto quickstatus]");
        } else if (user_line !== "" && !process.stdin.isTTY) {
             console.log("command: " + user_line);
        }

        const user_split = user_line.trim().split(/\s+/);
        const user_input = user_split[0];
        let reset_quickstatus = true;

        if (["status", "s"].includes(user_input)) {
            reset_quickstatus = false;
            const statuses = {};
            for (const qhit of contents) {
                if (user_split.length > 1) {
                    if (!new RegExp(user_split[1]).test(qhit)) continue;
                }
                const status = await get_status(qhit, auth);
                statuses[qhit] = status;
                console.log(qhit, JSON.stringify(status, null, 2));
            }
            if (!fs.existsSync("rundriver")) fs.mkdirSync("rundriver");
            fs.writeFileSync("rundriver/status.json", JSON.stringify(statuses, null, 2));

        } else if (["finalize", "f"].includes(user_input)) {
            console.log("it's called 'write' now (to avoid confusion over what it does)");
        } else if (["list", "l"].includes(user_input)) {
            const available_models = await list_models();
            console.log("available models:", available_models);
            console.log("models in current config:", models);
        } else if (["cw", "clearwritten"].includes(user_input)) {
            let iqsub = null;
            if (user_split.length > 1) iqsub = user_split[1];

            const new_written = {};
            if (iqsub) {
                for (const [iq, state] of Object.entries(written)) {
                    if (new RegExp(iqsub).test(iq)) {
                        console.log(iq, "cleared");
                    } else {
                        console.log(iq, "written");
                        new_written[iq] = written[iq];
                    }
                }
            }
            written = new_written;
        } else if (["stop"].includes(user_input)) {
            if (user_split.length < 2) {
                console.log("must specify iq and optionally model");
                continue
            }
            const iq = user_split[1];
            let stop_models;
            if (user_split.length > 2) {
                stop_models = [user_split[2]];
            } else {
                stop_models = models;
            }
            await stop(iq, auth, stop_models);
        } else if (["tag", "t"].includes(user_input)) {
            const this_tag_config = JSON.parse(JSON.stringify(tag_config));
            let iqsub = null;
            if (user_split.length > 1) iqsub = user_split[1];

            if (user_split.length > 2) {
                const model = user_split[2];
                if (this_tag_config.jobs) {
                    this_tag_config.jobs = this_tag_config.jobs.filter(job => job.model === model);
                    if (this_tag_config.jobs.length === 0) {
                        console.log(`Model '${model}' not found in config. Available models: ${models}`);
                        continue;
                    }
                }
            }

            const contentsub = contents.filter(x => iqsub === null || new RegExp(iqsub).test(x));
            await tag(contentsub, auth, argv.assets, this_tag_config, start_time, end_time);

        } else if (user_input.startsWith("+") || user_input.startsWith("-")) {
            let val = parseFloat(user_input);
            val = val * 60;

            if (end_time === null) {
                end_time = start_time + val;
            } else {
                end_time = end_time + val;
                start_time = start_time + val;
            }

            console.log(`[${start_time}-${end_time}] [${formatTime(start_time)} - ${formatTime(end_time)}]`);

        } else if (user_input === "qs") {
            reset_quickstatus = false;
            if (user_split.length > 1 && user_split[1] === "watch") {
                quickstatus_watch = ["qs", ...user_split.slice(2)].join(" ");
                console.log("quickstatus on, command: " + quickstatus_watch);
            } else if (user_split.length > 1 && user_split[1] === "off") {
                console.log("quickstatus off");
                quickstatus_watch = null;
            } else {
                const filter = user_split.slice(1).join(" ");
                for (const qhit of contents) {
                    await quick_status(auth, qhit, filter);
                }
            }
        } else if (["reverse"].includes(user_input)) {
            contents.reverse();
            console.log("First element:", contents[0]);
        } else if (["write", "w"].includes(user_input)) {
            let contentsub = contents;
            if (user_split.length > 1) contentsub = user_split.slice(1);
            await write_all(contentsub, argv.config, argv.commit, false);
        } else if (["forcewrite"].includes(user_input)) {
            let contentsub = contents;
            if (user_split.length > 1) contentsub = user_split.slice(1);
            await write_all(contentsub, argv.config, argv.commit, true);
        } else if (["agg", "aggregate"].includes(user_input)) {
            let contentsub = contents;
            if (user_split.length > 1) contentsub = user_split.slice(1);
            for (const qhit of contentsub) {
                await aggregate(qhit, argv.config, argv.commit);
            }
        } else if (["quit", "exit"].includes(user_input)) {
          break
        } else if (["h", "help"].includes(user_input)) {
            help();
        } else if (user_input === "") {
            reset_quickstatus = false;
        } else {
            reset_quickstatus = false;
            console.log(`Invalid command: ${user_input}`);
        }

        if (reset_quickstatus && quickstatus_watch) {
            quickstatus_watch = null;
            console.log("[auto quickstatus turned off]");
        }
    }

    console.log("Exiting");
    process.exit(0);
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
