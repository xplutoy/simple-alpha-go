# -*- coding: utf-8 -*-
# @Time  : 2019/3/26 20:55
# @Author : yx
# @Desc : ==============================================
# 使用建议，debug级别日志可以在各层函数内使用，info级别日志，尽量在外层接口中使用

import logging.config

config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s',
        },
        'terminal': {
            'format': '%(message)s',
        },
        # 其他的 formatter
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'terminal',
            # 'encoding': 'utf-8'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logging.log',
            'level': 'DEBUG',
            'formatter': 'simple',
            'encoding': 'utf-8'
        },
        # 其他的 handler
    },
    'loggers': {
        'root': {
            # 既有 console Handler，还有 file Handler
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
        # 其他的 Logger
    }
}
logging.config.dictConfig(config)

logger = logging.getLogger("root")
# logging.disable()
