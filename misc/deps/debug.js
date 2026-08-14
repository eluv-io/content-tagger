const { promisify } = require('util')
const { exec: exec_nonpromise } = require('child_process')
const exec = promisify(exec_nonpromise)


const defaultOptions = { "depth": null, iterableLimit: 200 }
let debugOn = false

function debugDir(obj, options = defaultOptions) {
    if (debugOn) console.dir(obj, options)                
}

function setDebug(on) {
    debugOn = on
    if (!console.oldDebug) console.oldDebug = console.debug
    if (!debugOn) {
        console.debug = () => {}
    }
    else {
        console.debug = console.oldDebug
    }
}

// not exactly debug, but close enough
async function gitRev() {
    try {
        const { stdout } = await exec("git rev-parse HEAD")
        return stdout.toString().trim()
    } catch {
        return "unknown"
    }
}

async function gitStatusCount() {
    try {
        const { stdout } = await exec("git status --porcelain");
        const lines = stdout.split('\n').filter(line => line);
        const counts = { untracked: 0, modified: 0, staged: 0 };

        lines.forEach(line => {
            const status = line.slice(0, 2).trim();
            if (status === '??') counts.untracked++;
            else if (status.startsWith('M')) counts.modified++;
            else if (status.startsWith('A') || status.startsWith('D')) counts.staged++;
        });

        return counts;
    } catch {
        return { untracked: 0, modified: 0, staged: 0 };
    }
}

async function gitRemote() {
    try {
        const { stdout } = await exec("git config --get remote.origin.url");
        return stdout.toString().trim();
    } catch {
        return "unknown";
    }
}

async function gitCurrentBranch() {
    try {
        const { stdout } = await exec("git rev-parse --abbrev-ref HEAD");
        return stdout.toString().trim();
    } catch {
        return "unknown";
    }
}

async function gitInfoblob() {
    const [statusCount, rev, remoteUrl, branch] = await Promise.all([
        gitStatusCount(),
        gitRev(),
        gitRemote(),
        gitCurrentBranch()
    ]);

    return {
        statusCount,
        rev,
        remoteUrl,
        branch
    };
}

module.exports = { debugDir, setDebug, gitRev, gitStatusCount, gitRemote, gitCurrentBranch, gitInfoblob };