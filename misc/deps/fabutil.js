#!/usr/bin/env false

const { ElvClient } = require("@eluvio/elv-client-js");
const { debugDir } = require("./debug.js");
const fs = require("fs").promises
const path = require("path")

function countItems(...items) {
    let count = 0
    for (const item of items) {
        count += Object.keys(item || {}).length
    }
    return count
}

/// Monkeypatching for client (caching, state channel auth)
async function cacheTokens(cacheFile, minimumNew = 0) {
    if (this.countTokens() < this.cachedTokens + minimumNew) {
        console.debug(`Not enough new tokens to cache (already cached ${this.cachedTokens}, total tokens ${this.countTokens()}, need ${minimumNew} new)`)
        return this
    }
    const tokenData = {
        accessTransactions: this.authClient.accessTransactions,
        modifyTransactions: this.authClient.modifyTransactions,
        channelContentTokens: this.authClient.channelContentTokens,
    }
    const dir = path.dirname(cacheFile)
    await fs.mkdir(dir, { recursive: true })
    const now = Date.now()
    await fs.writeFile(cacheFile + now, JSON.stringify(tokenData, null, 2), "utf8")
    await fs.rename(cacheFile + now, cacheFile)
    console.debug(`Cached tokens to ${cacheFile}`)
    this.cachedTokens = this.countTokens()
    return this
}

async function loadCachedTokens(cacheFile) {
    let tokenData = {}
    try {
        const data = await fs.readFile(cacheFile, "utf8")
        tokenData = JSON.parse(data)
        this.authClient.accessTransactions = { ...tokenData.accessTransactions, ...this.authClient.accessTransactions }
        this.authClient.modifyTransactions = { ...tokenData.modifyTransactions, ...this.authClient.modifyTransactions }
        this.authClient.channelContentTokens = { ...tokenData.channelContentTokens, ...this.authClient.channelContentTokens }
        console.debug(`Loaded cached tokens from ${cacheFile}`)
    } catch (err) {
        console.debug(`No cached tokens found at ${cacheFile}: ${err}`)
    }
    this.cachedTokens = countItems(tokenData.accessTransactions, tokenData.modifyTransactions, tokenData.channelContentTokens)
    return this
}

function countTokens() {
    return countItems(this.authClient.accessTransactions, this.authClient.modifyTransactions, this.authClient.channelContentTokens)
}

function alwaysAttemptStateChannelAuth(attempt = true) {

    this.authClient.alwaysAttemptStateChannelAuthSetting = attempt

    if (this.authClient.patchedForAlwaysAttemptStateChannelAuth) return this
    this.authClient.patchedForAlwaysAttemptStateChannelAuth = true

    const originalFunc = this.authClient.AuthorizationToken.bind(this.authClient)
    this.authClient.AuthorizationToken = async function(params) {

        console.debug(`AuthorizationToken called with params: ${JSON.stringify(params)}`)
        
        /* in here, this = authClient */
        let tryForcedStateChannel = this.alwaysAttemptStateChannelAuthSetting
        if (params.skipChannelAuth) tryForcedStateChannel = false
        if (params.update) tryForcedStateChannel = false
        if (params.skipChannelAuth) tryForcedStateChannel = false
        if (params.objectId == null) tryForcedStateChannel = false // (will try "normally")
        if (params.channelAuth) tryForcedStateChannel = false // (will try "normally")

        let result = null

        if (tryForcedStateChannel) {
            try {
                result = await originalFunc({...params, channelAuth: true})
            } catch (err) {
                console.debug(`forceStateChannelAuth caught error, retrying without channelAuth: ${err}`)
            }
        }
        if (!result) result = await originalFunc(params)
        console.debug(`Authorization token obtained`)

        return result
    }
    return this
}

function monkeypatchClient(client) {
    client.cacheTokens = cacheTokens.bind(client)
    client.loadCachedTokens = loadCachedTokens.bind(client)
    client.countTokens = countTokens.bind(client)
    client.alwaysAttemptStateChannelAuth = alwaysAttemptStateChannelAuth.bind(client)
    client.cachedTokens = 0
    return client
}
/// End of Monkeypatching for client

async function makeFabricClient() {
    const network = process.env.FABRIC_NETWORK || "main"
    console.debug(`makeFabricClient -- network ${network}`)

    const client = await ElvClient.FromNetworkName({networkName: network})
    const wallet = client.GenerateWallet();
    const signer = wallet.AddAccount({
      privateKey: process.env.PRIVATE_KEY,
    });
    client.SetSigner({ signer });
    return monkeypatchClient(client)
}

async function paginate(getPageFunc, nextPageFunc, pageStart = 0) {
    let allItems = []
    
    while (pageStart != null) {
        console.debug("before getPageFunc", pageStart)
        const response = await getPageFunc(pageStart)
        debugDir(response)
        allItems = allItems.concat(response.items)
        pageStart = nextPageFunc(response)
    }
    
    return allItems
}

async function trapCommonErrors(promise) {
    try {
        return await promise
    } catch (err) {
        return handleCommonErrors(err)
    }
}

function handleCommonErrors(err, contentId = null) {
    console.debug("-----------")
    debugDir(err)
    if (err.message.includes("Access denied")) {
        console.debug("Access denied to metadata" + (contentId ? " for contentId " + contentId : ""))
        return { ACCESS_DENIED: true }
    }
    throw err
}

function defaultOkToCache(data) {
    if (!data) return false
    if (typeof data === "object") return !(data.NO_CACHE == true);
    return true
}


async function cachedJson(cacheFile, dataFunctionOrObject, cacheMode = null, okToCacheFunc = defaultOkToCache) {

    if (!cacheMode) cacheMode = process.env.CACHE || "normal"
    console.debug("CacheMode " + cacheMode)

    try {
        if (cacheMode != "refresh" && cacheMode != "inject") {
            const data = await fs.readFile(cacheFile, "utf8")
            console.debug(`got cached data from ${cacheFile}`)
            const ret = JSON.parse(data)            
            if (typeof ret === "object") {
                delete ret.NO_CACHE
                ret.FROM_CACHE = true
            }
            return ret
        }
        else {
            throw new Error("force refresh")
        }
    } catch (err) {
        if (err.code == "ENOENT") {
            err = `${cacheFile} not found`
        }
        else if (err.message == "force refresh") {            
            err = "force refresh requested"
        }
        else {
            err = `error loading/parsing ${cacheFile}: ${err}`
        }

        console.debug(`fetch data from function - ${cacheFile} invalid: ${err}`)
        if (cacheMode == "cache-only") return null
      
        const data = (typeof dataFunctionOrObject == 'function') ? await dataFunctionOrObject() : dataFunctionOrObject
      
        if (typeof data === "object") data.FROM_CACHE = false
        const skipCache = (cacheMode == "read-only") || !okToCacheFunc(data)
                
        if (cacheMode == "read-only") {
            console.debug(`cacheMode read-only skipping cache write for ${cacheFile}`)
        }
        else if (skipCache) {
            console.debug(`data not ok to cache, skipping cache write for ${cacheFile}`)
        }

        delete(data.NO_CACHE)

        if (!skipCache) {
            const dir = path.dirname(cacheFile)
            await fs.mkdir(dir, { recursive: true })
            await fs.writeFile(cacheFile + ".tmp", JSON.stringify(data, null, 2), "utf8")
            await fs.rename(cacheFile + ".tmp", cacheFile)
            console.debug(`cached data to ${cacheFile}`)
        }

        return data
    }
}

function isNotNull(value) {
    return value !== null && value !== undefined
}

async function cachedBuffer(cacheFile, getDataFunc, cacheMode = null, okToCacheFunc = isNotNull) {

    if (!cacheMode) cacheMode = process.env.CACHE || "normal"
    try {
        if (cacheMode != "refresh") {
            const data = await fs.readFile(cacheFile)
            console.debug(`got cached buffer data from ${cacheFile}`)
            return data
        }
        else {
            throw new Error("force refresh")
        }
    }
    catch (err) {
        console.debug(`fetch buffer data from function - ${cacheFile} not found or invalid: ${err}`)
        if (cacheMode == "cache-only") {
            console.debug(`modifier/CACHE set to cache-only, skipping evaluation to fill ${cacheFile}`)
            return null
        }
        const data = await getDataFunc()
        if (cacheMode == "read-only") {
            console.debug(`modifier/CACHE set read-only, skipping cache write for ${cacheFile}`)
            return data
        }
        if (!okToCacheFunc(data)) {
            console.debug(`data not ok to cache, skipping cache write for ${cacheFile}`)
            return data
        }
        const dir = path.dirname(cacheFile)
        await fs.mkdir(dir, { recursive: true })
        await fs.writeFile(cacheFile, data)
        return data
    }
}

async function getLibContents(client, lib) {
    return paginate(
        async (page) => { 
            const res = await client.ContentObjects({
                libraryId: lib, 
                filterOptions: {
                    start: page || null,
                    select: "public/name"
                }
            })
            res.items = res.contents 
            return res
        },
        (res) => { return res.paging.next != res.paging.current ? res.paging.next : null }
    )
}



function extractFilePaths(data, directory, type) {
  //type: track, tracks, or overlay
  const directoryData = data[directory];
  if (!directoryData) {
    console.debug("Directory not found in the provided data");
    return [];
  }

  // Filter to get only files (ignore the directory key itself)
  const fileNames = Object.keys(directoryData).filter(
    (key) => key !== "." && key.includes(type)
  );
  return fileNames.map((fileName) => `${directory}/${fileName}`);
}


module.exports = {
  countItems,
  monkeypatchClient,
  makeFabricClient,
  paginate,
  trapCommonErrors,
  handleCommonErrors,
  handleAccessDenied: handleCommonErrors,
  cachedJson,  
  cachedBuffer,
  getLibContents,
  extractFilePaths
}


