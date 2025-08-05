from .option_kitti import KITTI_Config_
from .option_unet_nusc import NUSC_Config
from .option_nusc_box_layout import NUSC_Box_Layout_Config
from .option_nusc_box_layout_v1 import NUSC_Box_Layout_V1_Config
from .option_nusc_box_layout_v2 import NUSC_Box_Layout_V2_Config
from .option_nusc_box_layout_v3 import NUSC_Box_Layout_V3_Config
from .option_nusc_box_layout_v4 import NUSC_Box_Layout_V4_Config
from .option_nusc_box_layout_v5 import NUSC_Box_Layout_V5_Config
from .option_nusc_box_layout_v6 import NUSC_Box_Layout_V6_Config

from .option_meanflow_nusc import MeanFlow_NUSC_Config
from .option_nusc_layout import NUSC_Layout_Config
from .option_dit_nusc import NUSC_HDIT_Config
from .option_nusc_auto_reg import NUSC_Auto_Reg_Config
from .option_nusc_auto_reg_v2 import NUSC_Auto_Reg_V2_Config
from .option_nusc_object import NUSC_Object_Config
__all__ = {
    "kitti-360": KITTI_Config_,
    "nuscenes-unet-uncond": NUSC_Config,
    "nuscenes-hdit-uncond": NUSC_HDIT_Config,
    "nuscenes-auto-reg": NUSC_Auto_Reg_Config,
    'nuscenes-auto-reg-v2': NUSC_Auto_Reg_V2_Config,
    'nuscenes-box-layout': NUSC_Box_Layout_Config,
    'nuscenes-box-layout-v1': NUSC_Box_Layout_V1_Config,
    'nuscenes-box-layout-v2': NUSC_Box_Layout_V2_Config,
    'nuscenes-box-layout-v3': NUSC_Box_Layout_V3_Config,
    'nuscenes-box-layout-v4': NUSC_Box_Layout_V4_Config,
    'nuscenes-box-layout-v5': NUSC_Box_Layout_V5_Config,
    'nuscenes-box-layout-v6': NUSC_Box_Layout_V6_Config,
    'meanflow-nusc': MeanFlow_NUSC_Config,
    'nuscenes-layout': NUSC_Layout_Config,
    'nuscenes-object': NUSC_Object_Config,
}