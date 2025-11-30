## this python file shows how to send input to a running container
## that has been launched with the python podman client

## VERY IMPORTANT NOTE
##
## the container must be launched with
##    stdin_open=True
## in the container.create() call in order for this to work; otherwise the
## container process gets /dev/null as stdin and you can't write to it

## the meat of this is the function open_container_stdin
## it returns a socket that is connected to the stdin of the container process

## if desired the socket can be converted to a file like so
##
##    container_stdin_socket = open_container_stdin(container)
##    container_stdin_file = container_stdin_socket.makefile('w')
##    
## and written to with print(), but make sure to flush
##
##    print("write to container via print", file = container_stdin_file, flush = True)

## once this socket is closed, stdin of the container process will be closed
## and reading beyond the data sent will get EOF on stdin

## (long term if we want the containers to persist across the manager,
## we could potentially do this with a named pipe connected up to stdin in such a way
## that it would keep trying to read from it...)

## for this sample code to work you will need to do:
##
##    podman pull ubuntu
##
## to make sure we have the ubuntu image fetched

## imports not specifically related to this code:
from loguru import logger
from podman import PodmanClient
import time

############# PODMAN OPEN STDIN SOCKET FUNCTION #############
############# PODMAN OPEN STDIN SOCKET FUNCTION #############
############# PODMAN OPEN STDIN SOCKET FUNCTION #############

## these imports are needed specifically for the function
import socket
from urllib.parse import unquote

## this function should be able to be dropped into the job manager
## (or could even be monkeypatched into the podman container class)

## note that this assumes podman is on the same machine, and accessible
## via unix socket on this machine (typical for eluvio usecases)
def open_container_stdin(container):    
    
    ## assume podman is on the same machine via unix socket
    consocket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    consocket.connect(unquote(container.client.base_url.netloc))

    ## we have to do this manually, because once the podman socket server accepts the POST
    ## it then converts the socket into a "raw" socket for writing directly to the container's stdin
    msg = f"POST /v5.4.0/libpod/containers/{container.id}/attach?stdin=1&stdout=0&stderr=0 HTTP/1.0\r\n\r\n".encode()
    consocket.sendall(msg)

    ## response looks like this: 'HTTP/1.1 200 OK\r\nContent-Type: application/vnd.docker.raw-stream\r\n\r\n'
    response = consocket.recv(4096)
    
    logger.debug(f"socket response: {response}")
    
    ## make sure we successfully opened the connection
    ## be slightly flexible but otherwise pretty strict
    if response[0:15] != bytes("HTTP/1.1 200 OK", "utf-8") and response[0:15] != bytes("HTTP/1.0 200 OK", "utf-8"):
        raise Exception(f"Did not successfully open stdin for container: {response}")
    
    return consocket


############# SIMPLE EXAMPLE #############

## this example uses the stock ubuntu image to run "cat -n"
## (this just echoes the input but printing line numbers in front of the input)a
##
## we start the ubuntu container with "cat -n" as the command
## and specifying stdin_open=True which is required to write to stdin
##
## we then write lines into the container including a static string, the count, and the current time
## these get written approx 10 per second
##
## we also read the logs of the container once per second, to get the echoed results with the line number
##
## after writing 45 lines, we close the socket
## this results in closing of stdin, which the "cat" process reads as EOF
## then the container exits
##
## we read any final logs and get the container exit code

with PodmanClient() as podman_client:

    ## run a container running "cat -n"
    command = [ "cat", "-n" ]
    
    ## this is for testing exit status
    command_with_exit_status_1 = [ "cat", "-n", "nonexistent" ]
    
    kwargs = {
        "image": "docker.io/library/ubuntu",
        "command": command,
        "mounts": [{ "source": "/tmp", "target": "/tmp/host", "type": "bind" }],
        "remove": True, 
        "network_mode": "host",
        "stdin_open": True,  # <<<<<<<<<<<<<<<---------- this is required
        "stderr": True,
        "stdout": True,
        "devices": ["nvidia.com/gpu=0"]  ## assume 1 GPU
    }

    logger.debug("creating container...")
    container = podman_client.containers.create(**kwargs)

    logger.debug(f"starting container.  (container id {container.id})")
    container.start()
    
    logger.debug("waiting for container to progress beyond creation...")
    while container.status == "created":
        container.reload()

    logger.debug("container is running, opening socket")    
    container_stdin_socket = open_container_stdin(container)
    
    logger.debug(f"Opened socket: {container_stdin_socket}")

    ## weird time sync thing, not needed in prod code
    ## just to force there to be some logs to read after container stop

    now = int(time.time())
    while now == int(time.time()):
        time.sleep(.03)
        
    count = 0
    lastread = 0

    ## as long as the container is running, check it for logs once a second
    ## and write some lines into it
    while container.status == "running":
        container.reload()

        count = count + 1

        if count <= 45:
            ## write a line into the container
            message = f"HELLO!  count: {count}\t  at the beep, the time will be: {time.time()}\n"

            ## NOTE!! this can block, if the container is not reading from its standard input for some reason
            container_stdin_socket.sendall(message.encode())
            
        elif count == 46:
            ## we've written 45 lines in, let's close the socket
            ## the container will see this as end-of-file on standard input (and likely exit soon)
            logger.debug("CLOSING SOCKET NOW");
            container_stdin_socket.close()
            
        ## read the logs no more than once a second        
        now = int(time.time())
        if now == lastread:
            ## checked logs too recently, just sleep a little bit instead of reading logs
            time.sleep(.1)
        else:
            logs = container.logs(stream=False, stderr=True, stdout=True, since=lastread, until=now)
            lastread = now
            
            for log in logs:
                logger.info("log from container: " + log.decode("utf-8").replace("\n", ""))

                
    logger.debug(f"container is no longer running, it is in status: {container.status}")
    logger.debug("reading final logs")
    
    now = int(time.time())
    while now == lastread:
        now = int(time.time())
        time.sleep(.05)

    logs = container.logs(stream=False, stderr=True, stdout=True, since=lastread)
    for log in logs:
        logger.info("log from container: " + log.decode("utf-8").replace("\n", ""))

    exitcode = container.wait()
    logger.info(f"container exit code: {exitcode}")
        
