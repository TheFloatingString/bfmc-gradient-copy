import sys
sys.path.append(".")
from multiprocessing import Queue, Event
import logging

# === process imports ===
from src.hardware.serialhandler.processSerialHandler import processSerialHandler
from src.gateway.processGateway import processGateway

# === setup ===
all_processes = list()
queue_list = {
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue()
        }

queue_list["General"].put({"action":"speed","value":1.0,"Onwer":"Laurence"})
logging = logging.getLogger()
print(queue_list["General"])
process_gateway = processGateway(queue_list, logging)
all_processes.append(process_gateway)
process_serial_handler = processSerialHandler(queue_list, logging)
all_processes.append(process_serial_handler)

for process in all_processes:
    print("starting process:", process)
    process.daemon = True
    process.start()
    process.stop()
    process.join()
    print("end of process.")
