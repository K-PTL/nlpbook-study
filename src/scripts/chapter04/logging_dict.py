import logging.config

# based on -> https://qiita.com/__init__/items/91e5841ed53d55a7895e
logging.config.dictConfig({
    'version': 1,
    "disable_existing_loggers": False, # allow to overwrite

    # フォーマットの設定
    'formatters': {
        'customFormat': {
            'format': '%(asctime)s - %(name)-12s - %(levelname)-8s - %(message)s'
        },
        "logFileFormatter": {
            "format": "%(asctime)s|%(levelname)-8s|%(name)s|%(funcName)s|%(message)s"
        }
    },
    # ハンドラの設定
    'handlers': {
        'customStreamHandler': {
            'class': 'logging.StreamHandler',
            'formatter': 'customFormat',
            'level': logging.DEBUG
        },

        "logFileHandler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "logFileFormatter",
            "filename": "./log/app.log", # 先にlogディレクトリを作成しないとエラーを吐く
            "mode": "w",
            "encoding": "utf-8"
        }
    },

    # ロガーの対象一覧
    'root': {
        'handlers': ['customStreamHandler', "logFileHandler"],
        'level': logging.DEBUG
    },

    'loggers': {
        'stopwatchLogging': {
            'handlers': ['customStreamHandler', "logFileHandler"],
            'level': logging.DEBUG,
            'propagate': 0
        },
        'loadLogging': {
            'handlers': ['customStreamHandler', "logFileHandler"],
            'level': logging.DEBUG,
            'propagate': 0
        },
        'preproLogging': {
            'handlers': ['customStreamHandler', "logFileHandler"],
            'level': logging.WARNING,
            'propagate': 0
        },
        'trnevLogging': { # TRain_aNd_EVal
            'handlers': ['customStreamHandler', "logFileHandler"],
            'level': logging.DEBUG,
            'propagate': 0
        }
    }
})
