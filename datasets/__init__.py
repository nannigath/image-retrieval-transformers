from .cub200 import Cub200Dataset
from .sop import SOPDataset
from .inshop import InShopDataset
from .custom import TuplesDataset


def get_dataset(args):

    """returns train, query and gallery dataset"""

    train, query, gallery = None, None, None

    if args.dataset == 'cub200':
        train   = Cub200Dataset(args.data_path, split="train")
        query   = Cub200Dataset(args.data_path, split="test")

    if args.dataset == 'sop':
        train   = SOPDataset(args.data_path, split="train")
        query   = SOPDataset(args.data_path, split="test")

    if args.dataset == 'inshop':
        train   = InShopDataset(args.data_path, split="train")
        query   = InShopDataset(args.data_path, split="query")
        gallery = InShopDataset(args.data_path, split="gallery")
    
    if args.dataset == 'custom':
        if args.task == 'purticular':
            train   = TuplesDataset(args.data_path, split="train")
            query   = TuplesDataset(args.data_path, split="val")

    return train, query, gallery
