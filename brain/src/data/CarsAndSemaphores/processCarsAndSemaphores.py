# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
if __name__ == "__main__":
    import sys

    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.data.CarsAndSemaphores.threads.threadCarsAndSemaphores import (
    threadCarsAndSemaphores,
)


class processCarsAndSemaphores(WorkerProcess):
    """This process will receive the location of the other cars and the location and the state of the semaphores.

    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
    """

    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging=False):
        self.queuesList = queueList
        self.logging = logging
        super(processCarsAndSemaphores, self).__init__(self.queuesList)

    # ===================================== STOP ==========================================
    def stop(self):
        """Function for stopping threads and the process."""
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processCarsAndSemaphores, self).stop()

    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processCarsAndSemaphores, self).run()

    # ===================================== INIT TH ======================================
    def _init_threads(self):
        """Create the thread and add to the list of threads."""
        CarsSemTh = threadCarsAndSemaphores(self.queuesList)
        self.threads.append(CarsSemTh)


# =================================== EXAMPLE =========================================
#             ++    THIS WILL RUN ONLY IF YOU RUN THE CODE FROM HERE  ++
#                  in terminal:    python3 processCarsAndSemaphores.py


from threading import Thread
from multiprocessing import Queue
import time

s1 = None
s3 = None

def p_get_s1():
    global s1
    return s1

def p_get_s3():
    global s3
    return s3

def p_run():
    global s1
    global s3

    # '''
    queueList = {
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
    }

    allProcesses = list()
    process = processCarsAndSemaphores(queueList)
    process.start()

    time.sleep(3)
    while True:
        tmp = queueList["General"].get()
        # print(tmp)
        # if s1 -> update s1
        if tmp['msgID']==2 and tmp['msgValue']['id']==1:
            s1 = tmp['msgValue']['state']
        # if s3 -> update s3
        if tmp['msgID']==2 and tmp['msgValue']['id']==3:
            s3 = tmp['msgValue']['state']

    process.stop()
    # '''

    s1 = 'green'
    s3 = 'green'

print("hi!")
t = Thread(target=p_run)
t.start()
print('hi!')
# p_run()
