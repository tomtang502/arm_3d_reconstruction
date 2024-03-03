import pybullet as p

joint_info_name = ("""jointIndex
jointName
jointType
qIndex
uIndex
flags
jointDamping
jointFriction
jointLowerLimit
jointUpperLimit
jointMaxForce
jointMaxVelocity
linkName
jointAxis
parentFramePos
parentFrameOrn
parentIndex
""".split())

def joint_inf_prt(bid, jidx):
    print("_"*30+f"Joint idx: {jidx}"+"_"*30)
    info = p.getJointInfo(bid, jidx)
    for i in range(len(joint_info_name)):
        print(f"{joint_info_name[i]} : {info[i]}")
    print("-"*30+"End Split"+"-"*30)

def info_prt(s, name = ''):
    print("_"*60)
    if len(name) == 0:
        print(s)
    else:
        print(f"{name} : {s}")
    print("-"*60)