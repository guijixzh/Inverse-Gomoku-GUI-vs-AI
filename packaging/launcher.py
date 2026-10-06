"""PyInstaller 打包启动器(无 torch 分发版)。

打包命令(在仓库根目录):
    pyinstaller packaging/Antifive.spec --noconfirm
"""

import multiprocessing

from antifive.gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
