import json
import java.io
from org.apache.commons.io import IOUtils
from java.nio.charset import StandardCharsets
from org.apache.nifi.processor.io import StreamCallback
from org.apache.nifi.processors.script import ExecuteScript


class PyStreamCallback(StreamCallback):
    def __init__(self):
        pass

    def process(self, inputStream, outputStream):
        text = IOUtils.toString(inputStream, StandardCharsets.UTF_8)
        obj=json.loads(text)
        newObj = {
            "Source": "NiFi",
            "ID": "python",
            "Name": "test",
            "meta_data": obj['rating']['metric']['value']
        }
        outputStream.write(bytearray(json.dumps(newObj, indent=4).encode('utf-8')))


flowFile = session.get()
if flowFile != None:
    flowFile = session.write(flowFile, PyStreamCallback())
    session.transfer(flowFile, ExecuteScript.REL_SUCCESS)
else:
    pass