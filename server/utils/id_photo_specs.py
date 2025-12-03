"""
证件照规格数据库
包含各种场景下的证件照尺寸标准
"""

# 证件照规格数据库
# 格式: (宽_mm, 高_mm, 宽_px, 高_px)
# 像素按照 300 DPI 计算

ID_PHOTO_SPECS = {
    # ========== 常用标准 ==========
    "standard": {
        "category_name": "常用标准",
        "icon": "📏",
        "specs": {
            "1inch": {
                "name": "一寸",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white", "blue", "red"],
                "description": "最常用的证件照尺寸",
                "usage": "简历、学生证、工作证等"
            },
            "2inch": {
                "name": "二寸",
                "size_mm": (35, 49),
                "size_px": (413, 579),
                "common_bg": ["white", "blue", "red"],
                "description": "较大的标准尺寸",
                "usage": "毕业照、档案照等"
            },
            "small_1inch": {
                "name": "小一寸",
                "size_mm": (22, 32),
                "size_px": (260, 378),
                "common_bg": ["white", "blue"],
                "description": "略小于一寸",
                "usage": "部分证件使用"
            },
            "large_1inch": {
                "name": "大一寸",
                "size_mm": (33, 48),
                "size_px": (390, 567),
                "common_bg": ["white", "blue", "red"],
                "description": "略大于一寸",
                "usage": "护照、签证等"
            },
            "small_2inch": {
                "name": "小二寸",
                "size_mm": (33, 48),
                "size_px": (390, 567),
                "common_bg": ["white", "blue"],
                "description": "略小于二寸",
                "usage": "部分证件"
            },
        }
    },
    
    # ========== 身份证件 ==========
    "identification": {
        "category_name": "身份证件",
        "icon": "🪪",
        "specs": {
            "id_card": {
                "name": "身份证",
                "size_mm": (26, 32),
                "size_px": (358, 441),
                "common_bg": ["white"],
                "description": "二代身份证照片",
                "usage": "办理身份证专用",
                "note": "白色背景，免冠正面照"
            },
            "household_register": {
                "name": "户口本",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white", "blue"],
                "description": "户口本登记照",
                "usage": "户口登记、迁移等"
            },
            "temp_id": {
                "name": "临时身份证",
                "size_mm": (26, 32),
                "size_px": (358, 441),
                "common_bg": ["white"],
                "description": "临时身份证照片",
                "usage": "身份证遗失期间使用"
            },
        }
    },
    
    # ========== 出入境证件 ==========
    "travel": {
        "category_name": "出入境证件",
        "icon": "✈️",
        "specs": {
            "passport": {
                "name": "护照",
                "size_mm": (33, 48),
                "size_px": (390, 567),
                "common_bg": ["white"],
                "description": "中国护照标准",
                "usage": "护照办理、更新",
                "note": "白色背景，6个月内近照"
            },
            "hk_macau_permit": {
                "name": "港澳通行证",
                "size_mm": (33, 48),
                "size_px": (390, 567),
                "common_bg": ["white"],
                "description": "往来港澳通行证",
                "usage": "港澳通行证办理"
            },
            "taiwan_permit": {
                "name": "台湾通行证",
                "size_mm": (33, 48),
                "size_px": (390, 567),
                "common_bg": ["white"],
                "description": "大陆居民往来台湾通行证",
                "usage": "台湾通行证办理"
            },
            "visa_us": {
                "name": "美国签证",
                "size_mm": (51, 51),
                "size_px": (600, 600),
                "common_bg": ["white"],
                "description": "美国签证专用",
                "usage": "美国签证申请",
                "note": "正方形，白色背景"
            },
            "visa_schengen": {
                "name": "申根签证",
                "size_mm": (35, 45),
                "size_px": (413, 531),
                "common_bg": ["white"],
                "description": "申根国家签证",
                "usage": "欧洲申根国签证"
            },
        }
    },
    
    # ========== 驾驶证件 ==========
    "driving": {
        "category_name": "驾驶证件",
        "icon": "🚗",
        "specs": {
            "driving_license": {
                "name": "驾驶证",
                "size_mm": (22, 32),
                "size_px": (260, 378),
                "common_bg": ["white"],
                "description": "机动车驾驶证",
                "usage": "驾照申领、换证",
                "note": "白色背景，免冠正面照"
            },
            "driver_qualification": {
                "name": "从业资格证",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white"],
                "description": "道路运输从业资格证",
                "usage": "货运、客运从业资格"
            },
        }
    },
    
    # ========== 考试报名 ==========
    "exam": {
        "category_name": "考试报名",
        "icon": "📝",
        "specs": {
            "civil_servant": {
                "name": "公务员考试",
                "size_mm": (35, 45),
                "size_px": (413, 531),
                "common_bg": ["white", "blue"],
                "description": "国家/地方公务员考试",
                "usage": "公考报名专用",
                "note": "近期免冠正面证件照"
            },
            "cet": {
                "name": "英语四六级",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["blue"],
                "description": "大学英语四六级考试",
                "usage": "CET-4/6 报名"
            },
            "ncee": {
                "name": "高考报名",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["blue"],
                "description": "普通高等学校招生考试",
                "usage": "高考报名"
            },
            "postgraduate": {
                "name": "研究生考试",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["blue"],
                "description": "全国硕士研究生招生考试",
                "usage": "考研报名"
            },
            "teacher_qualification": {
                "name": "教师资格证",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white", "blue"],
                "description": "教师资格证考试",
                "usage": "教资考试报名"
            },
            "cpa": {
                "name": "注册会计师",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white"],
                "description": "注册会计师考试",
                "usage": "CPA 考试报名"
            },
            "judicial_exam": {
                "name": "法律职业资格",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white"],
                "description": "国家统一法律职业资格考试",
                "usage": "司法考试报名"
            },
        }
    },
    
    # ========== 学历学位 ==========
    "education": {
        "category_name": "学历学位",
        "icon": "🎓",
        "specs": {
            "student_id": {
                "name": "学生证",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["blue", "red"],
                "description": "学生证件照",
                "usage": "学生证制作"
            },
            "graduation": {
                "name": "毕业照",
                "size_mm": (35, 49),
                "size_px": (413, 579),
                "common_bg": ["blue"],
                "description": "毕业证件照",
                "usage": "毕业证书、学位证"
            },
            "degree_verification": {
                "name": "学历认证",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["blue"],
                "description": "学历学位认证",
                "usage": "学信网认证等"
            },
        }
    },
    
    # ========== 职业资格 ==========
    "professional": {
        "category_name": "职业资格",
        "icon": "💼",
        "specs": {
            "health_cert": {
                "name": "健康证",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white"],
                "description": "健康证明",
                "usage": "从业健康证"
            },
            "work_permit": {
                "name": "工作证",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["blue", "red"],
                "description": "工作证件",
                "usage": "单位工作证"
            },
            "professional_cert": {
                "name": "职业资格证",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white"],
                "description": "各类职业资格证书",
                "usage": "职业技能证书"
            },
        }
    },
    
    # ========== 社保医疗 ==========
    "social_security": {
        "category_name": "社保医疗",
        "icon": "🏥",
        "specs": {
            "social_security": {
                "name": "社保卡",
                "size_mm": (26, 32),
                "size_px": (358, 441),
                "common_bg": ["white"],
                "description": "社会保障卡",
                "usage": "社保卡办理"
            },
            "medical_insurance": {
                "name": "医保卡",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white"],
                "description": "医疗保险卡",
                "usage": "医保卡办理"
            },
        }
    },
    
    # ========== 其他用途 ==========
    "others": {
        "category_name": "其他用途",
        "icon": "📋",
        "specs": {
            "resume": {
                "name": "简历照片",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white", "blue"],
                "description": "求职简历照片",
                "usage": "个人简历"
            },
            "marriage_cert": {
                "name": "结婚证",
                "size_mm": (40, 60),
                "size_px": (472, 708),
                "common_bg": ["red"],
                "description": "结婚登记照",
                "usage": "结婚登记",
                "note": "红色背景，双人合影"
            },
            "residence_permit": {
                "name": "居住证",
                "size_mm": (25, 35),
                "size_px": (295, 413),
                "common_bg": ["white"],
                "description": "居住证照片",
                "usage": "居住证办理"
            },
        }
    },
}

# 背景颜色预设
BG_COLOR_PRESETS = {
    "white": {
        "name": "白色",
        "hex": "#FFFFFF",
        "rgb": (255, 255, 255),
        "description": "最常用，适用于大多数证件"
    },
    "blue": {
        "name": "蓝色",
        "hex": "#438EDB",
        "rgb": (67, 142, 219),
        "description": "标准证件蓝，常用于考试报名"
    },
    "red": {
        "name": "红色",
        "hex": "#D9001B",
        "rgb": (217, 0, 27),
        "description": "标准证件红，用于特定证件"
    },
    "gray": {
        "name": "灰色",
        "hex": "#607D8B",
        "rgb": (96, 125, 139),
        "description": "深灰色背景"
    },
}

def get_all_specs():
    """获取所有规格，扁平化返回"""
    all_specs = {}
    for category_id, category_data in ID_PHOTO_SPECS.items():
        for spec_id, spec_info in category_data["specs"].items():
            all_specs[spec_id] = spec_info
    return all_specs

def get_spec_by_id(spec_id: str):
    """根据 ID 获取规格信息"""
    for category_data in ID_PHOTO_SPECS.values():
        if spec_id in category_data["specs"]:
            return category_data["specs"][spec_id]
    return None

def search_specs(keyword: str):
    """搜索规格"""
    results = []
    keyword = keyword.lower()
    for category_id, category_data in ID_PHOTO_SPECS.items():
        for spec_id, spec_info in category_data["specs"].items():
            # 搜索名称、描述、用途
            if (keyword in spec_info["name"].lower() or 
                keyword in spec_info["description"].lower() or 
                keyword in spec_info["usage"].lower()):
                results.append({
                    "id": spec_id,
                    "category": category_data["category_name"],
                    "category_icon": category_data["icon"],
                    **spec_info
                })
    return results
