view_to_cam_type = {
    "/camera/front/main": "FW",
    "/camera/front/narrow": "FN",
    "/camera/side/front/left": "FL",
    "/camera/side/front/right": "FR",
    "/camera/side/rear/left": "RL",
    "/camera/side/rear/right": "RR",
    "/camera/rear": "RN",
    "/camera/surrouding/left": "SLW",
    "/camera/surrouding/rear": "SRCW",
    "/camera/surrouding/right": "SRW",
    "/camera/surrouding/front": "SFW"
}


adas_cam_type_to_view = {
    "ADAS_Front120": "/camera/front/main",
    "ADAS_Front30": "/camera/front/narrow",
    "ADAS_FrontLeft": "/camera/side/front/left",
    "ADAS_FrontRight": "/camera/side/front/right",
    "ADAS_RearLeft": "/camera/side/rear/left",
    "ADAS_RearRight": "/camera/side/rear/right",
    "ADAS_Rear": "/camera/rear",
    "SVC_Front": "/camera/surrouding/front",
    "SVC_Left": "/camera/surrouding/left",
    "SVC_Right": "/camera/surrouding/right",
    "SVC_Rear": "/camera/surrouding/left",
}


navi_info_keys = dict(
    road_tags=["road_class", "road_type"] + ["main_action", "assist_action"],
    guide_group=["guide_main_action", "guide_assist_action", "guide_distance"],
    traffic_info=[
        "traffic_light_direction",
        "traffic_light_type",
        "traffic_light_countdown",
        "traffic_light_distance",
        "speedLimit",
    ],
)

lane_info_keys = [
    "recommend",
    "can_drive",
    "lane_type",
    "lane_direction",
    "lane_highlight_direction",
    "lane_change_type",
]

navi_cls_mapping = {
    "pad": 0,
    "tld": 1,
    "spd_limit": 2,
    "action_1": 3,
    "action_2": 4,
    "lane_info": 5,
    "lane_highline_info": 6
}


tld_to_lane_direction_mapping = [
    0b10000, # 0 left turn
    0b1000, # 1 left
    0b0100, # 2 straight
    0b0010 # 3 right
]

lane_direction_to_tld_id_mapping = {
    0b10000: 0,
    0b1000: 1,
    0b0100: 2,
    0b0010: 3,
    0b0001: 3,
}

MAIN_ACTION_MAPPING = {
    0: "无",
    1: "左转",                # 左转0x1
    2: "右转",                # 右转0x2
    3: "向左前方行驶",        # 向左前方行驶0x3
    4: "向右前方行驶",        # 向右前方行驶0x4
    5: "向左后方行驶",        # 向左后方行驶0x5
    6: "向右后方行驶",        # 向右后方行驶0x6
    7: "左转调头",            # 左转调头0x7
    8: "直行",                # 直行0x8
    9: "靠左",                # 靠左0x9
    10: "靠右",               # 靠右
    11: "进入环岛",           # 进入环岛
    12: "离开环岛",           # 离开环岛
    13: "减速",               # 减速
    14: "插入直行",           # 插入直行
    65: "进入建筑物",         # 进入建筑物
    66: "离开建筑物",         # 离开建筑物
    67: "电梯换层",           # 电梯换层
    68: "楼梯换层",           # 楼梯换层
    69: "扶梯换层",           # 扶梯换层
    70: "COUNT"               # 导航主动作最大个数
}

# assist action mapping
ASSIST_ACTION_MAPPING = {
    0: "无",
    1: "进入主路",
    2: "进入辅路",
    3: "进入高速",
    4: "进入匝道",
    5: "进入隧道",
    6: "进入中间岔道",
    7: "进入右岔路",
    8: "进入左岔路",
    9: "进入右转专用道",
    10: "进入左转专用道",
    11: "进入中间道路",
    12: "进入右侧道路",
    13: "进入左侧道路",
    14: "靠右进入辅路",
    15: "靠左进入辅路",
    16: "靠右进入主路",
    17: "靠左进入主路",
    18: "靠右进入右转专用道",
    19: "到达航道",
    20: "驶离轮渡",
    23: "沿当前道路行驶",
    24: "沿辅路行驶",
    25: "沿主路行驶",
    32: "到达出口",
    33: "到达服务区",
    34: "到达收费站",
    35: "到达途径地",
    36: "到达目的地",
    37: "到达充电站",
    48: "沿环岛左转",
    49: "绕环岛右转",
    50: "绕环岛直行",
    51: "绕环岛掉头",
    52: "小环岛不数个数",
    64: "复杂路口，走右边第一出口",
    65: "复杂路口，走右边第二出口",
    66: "复杂路口，走右边第三出口",
    67: "复杂路口，走右边第四出口",
    68: "复杂路口，走右边第五出口",
    69: "复杂路口，走左边第一出口",
    70: "复杂路口，走左边第二出口",
    71: "复杂路口，走左边第三出口",
    72: "复杂路口，走左边第四出口",
    73: "复杂路口，走左边第五出口",
    80: "进入调头专用路",
    90: "通过人行横道",
    91: "通过过街天桥",
    92: "通过地下通道",
    93: "通过广场",
    94: "通过公园",
    95: "通过扶梯",
    96: "通过直梯",
    97: "通过索道",
    98: "通过空中通道",
    99: "通过建筑物穿越通道",
    100: "通过行人道路",
    101: "通过游船路线",
    102: "通过观光车路线",
    103: "通过滑道",
    105: "通过阶梯",
    106: "通过斜坡",
    107: "通过桥",
    108: "通过轮渡",
    109: "通过地铁通道",
    112: "即将进入建筑",
    113: "即将离开建筑",
    114: "进入环岛",
    115: "离开环岛",
    116: "进入小路",
    117: "进入内部路",
    118: "进入左侧第二岔路",
    119: "进入左侧第三岔路",
    120: "进入右侧第二岔路",
    121: "进入右侧第三岔路",
    122: "进入加油站道路",
    123: "进入小区道路",
    124: "进入园区道路",
    125: "上高架",
    126: "走中间岔路上高架",
    127: "走最右侧岔路上高架",
    128: "走最左侧岔路上高架",
    129: "沿当前道路直行",
    130: "下高架",
    131: "走左侧道路上高架",
    132: "走右侧道路上高架",
    133: "上桥",
    134: "进停车场",
    135: "进立交桥",
    136: "进桥梁",
    137: "进地下通道",
    4096: "MAX"
}

kLaneTypeMap = {
    0: "无效车道",
    1: "普通车道",
    2: "公交车道",
    3: "公交车道文字",
    4: "可变车道",
    5: "HOV",
    6: "潮汐车道文字",
    7: "潮汐车道前行箭头",
    8: "潮汐车道叉号",
    9: "ETC (百度引擎)",
    10: "专用车道（高德引擎）",
    11: "..."
}

COLORS_ID = [
    (144, 238, 144),
    (178, 34, 34),
    (221, 160, 221),
    (0, 128, 0),
    (210, 105, 30),
    (220, 20, 60),
    (192, 192, 192),
    (255, 228, 196),
    (50, 205, 50),
    (139, 0, 139),
    (100, 149, 237),
    (138, 43, 226),
    (238, 130, 238),
    (255, 0, 255),
    (0, 100, 0),
    (127, 255, 0),
    (255, 0, 255),
    (255, 140, 0),
    (255, 239, 213),
    (199, 21, 133),
    (124, 252, 0),
    (147, 112, 219),
    (106, 90, 205),
    (176, 196, 222),
    (65, 105, 225),
    (173, 255, 47),
    (255, 20, 147),
    (219, 112, 147),
    (186, 85, 211),
    (199, 21, 133),
    (148, 0, 211),
    (255, 99, 71),
    (144, 238, 144),
    (255, 255, 0),
    (230, 230, 250),
    (128, 128, 0),
    (189, 183, 107),
    (255, 255, 224),
    (128, 128, 128),
    (105, 105, 105),
    (64, 224, 208),
    (205, 133, 63),
    (0, 128, 128),
    (72, 209, 204),
    (139, 69, 19),
    (255, 245, 238),
    (250, 240, 230),
    (152, 251, 152),
    (0, 255, 255),
    (135, 206, 235),
    (0, 191, 255),
    (176, 224, 230),
    (0, 250, 154),
    (245, 255, 250),
    (240, 230, 140),
    (245, 222, 179),
    (0, 139, 139),
    (143, 188, 143),
    (240, 128, 128),
    (102, 205, 170),
    (60, 179, 113),
    (46, 139, 87),
    (165, 42, 42),
    (178, 34, 34),
    (175, 238, 238),
    (255, 248, 220),
    (218, 165, 32),
    (255, 250, 240),
    (253, 245, 230),
    (244, 164, 96),
    (210, 105, 30),
]

def get_lane_direction_str(lane_direction):
    lane_direction_str = ""
    if lane_direction & 0x1:
        lane_direction_str += "↓"
    if (lane_direction >> 1) & 0x1:
        lane_direction_str += "→"
    if (lane_direction >> 2) & 0x1:
        lane_direction_str += "↑"
    if (lane_direction >> 3) & 0x1:
        lane_direction_str += "←"
    if (lane_direction >> 4) & 0x1:
        lane_direction_str += "↓"
    return lane_direction_str


road_arrow_to_lane_direction_mapping = {
    201: 0b01000,  # RoadSignClass_RoadArrow_Left
    202: 0b00010,  # RoadSignClass_RoadArrow_Right
    203: 0b01010, # RoadSignClass_RoadArrow_Left_Right
    204: 0b00100,  # RoadSignClass_RoadArrow_Straight
    205: 0b00110,  # RoadSignClass_RoadArrow_Straight_Right
    206: 0b01100,  # RoadSignClass_RoadArrow_Straight_Left
    207: 0b10000,  # RoadSignClass_RoadArrow_Only_Turn
    208: 0b11000,  # RoadSignClass_RoadArrow_Left_Turn
    209: 0b10100,  # RoadSignClass_RoadArrow_Straight_Turn
    210: 0b00100,  # RoadSignClass_RoadArrow_Left_Merge
    211: 0b00100,  # RoadSignClass_RoadArrow_Right_Merge
    213: 0b00000,  # RoadSignClass_RoadArrow_Forbid_Only_Sign
    214: 0b10110,  # RoadSignClass_RoadArrow_Forbid_Left_Sign
    215: 0b11100,  # RoadSignClass_RoadArrow_Forbid_Right_Sign
    216: 0b11010,  # RoadSignClass_RoadArrow_Forbid_Straight_Sign
    217: 0b01110,  # RoadSignClass_RoadArrow_Forbid_Turn_Sign
    220: 0b00110,  # RoadSignClass_RoadArrow_Forbid_Left_And_Turn_Sign
}

NaviLabelMapping = {
    0: "unknown",
    1: "navi",
    2: "not_navi",
    3: "highlight",
}