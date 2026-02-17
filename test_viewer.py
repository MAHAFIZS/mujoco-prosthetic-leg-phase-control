import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
    <worldbody>
        <geom type="box" size="0.1 0.1 0.1"/>
    </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

v = mujoco.viewer.launch_passive(m, d)
time.sleep(5)
v.close()
