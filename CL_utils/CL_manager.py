from .CL_state import CL_states, Enhance_COCO
from pathlib import Path
DATA_ROOT = Path('../dataset/voc2007/')

IMG_FILE_ROOT = DATA_ROOT / 'images'
TRAIN_DATA = DATA_ROOT / "annotations/voc2007_trainval.json"
TEST_DATA = DATA_ROOT / "annotations/voc2007_test.json"

IMAGE_TXT_ROOT = DATA_ROOT / 'images_paths'

class CL_manager(object):
    def __init__(self, scenario:list, use_all_label=False, test_replay=False):
        self.cl_states = CL_states(TRAIN_DATA, scenario)
        self.train_coco =  Enhance_COCO(TRAIN_DATA)
        self.test_coco = Enhance_COCO(TEST_DATA)
        self.test_replay = test_replay
        self.use_all_label = use_all_label

    def gen_data_dict(self, cur_state:int):
        """generate data dictionary (replace the yaml file, for example: VOC_2007.yaml)
        """

        data_dict = { 'names':self.cl_states[cur_state]['knowing_class']['name'],
                    'nc':self.cl_states[cur_state]['num_knowing_class'],
                    'train':IMAGE_TXT_ROOT / f'train_images_{self.cl_states.scenario}_{cur_state}.txt',
                    'val':IMAGE_TXT_ROOT / f'test_images_{self.cl_states.scenario}_{cur_state}.txt',
                    'test':IMAGE_TXT_ROOT / f'test_images_{self.cl_states.scenario}_{cur_state}.txt',
                    }
        if not Path(data_dict['train']).exists():
            lines = [str(IMG_FILE_ROOT / '{:06d}.jpg\n'.format(img_id)) for img_id in self.train_coco.get_imgs_by_cats(self.cl_states[cur_state]['new_class']['id'])]
            if self.test_replay:
                from CL_replay import read_exemplar
                lines += [str(IMG_FILE_ROOT / '{:06d}.jpg\n'.format(img_id)) for img_id in read_exemplar()]
            with open(data_dict['train'], 'w') as f:
                f.writelines(lines)
        if not Path(data_dict['test']).exists():
            lines = [str(IMG_FILE_ROOT / '{:06d}.jpg\n'.format(img_id)) for img_id in self.test_coco.get_imgs_by_cats(self.cl_states[cur_state]['knowing_class']['id'])]
            with open(data_dict['test'], 'w') as f:            
                f.writelines(lines)
        return data_dict

    def gen_yolo_lables(self, cur_state:int):
        """According current state to generate the yolo format's lables
        """
        def convert_box(size, box):
            dw, dh = 1. / size[0], 1. / size[1]
            xlt, ylt, w, h = box[0], box[1], box[2], box[3] #lt:left top
            x , y = xlt + (w / 2.0), ylt + (h / 2.0)
            return x * dw, y * dh, w * dw, h * dh

        def gen_labels(target_path:Path, img_ids:list, coco_obj:Enhance_COCO, seen_ids:list, start_idx:int):
            for img_id in img_ids:
                file_name = "{:06d}.txt".format(img_id)
                info = coco_obj.loadImgs(img_id)[0]
                w, h = info['width'], info['height']
                anns = coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=img_id))
                lines = []
                for ann in anns:
                    if ann['category_id'] not in seen_ids:
                        continue
                    cls_id = seen_ids.index(ann['category_id']) + start_idx
                    bb = convert_box((w, h),ann['bbox'])
                    lines.append(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')
                
                #file_list.append(target_path / file_name)
                with open(target_path / file_name, 'w') as f:
                    f.writelines(lines)

        #Training Labels
        target_path = DATA_ROOT / 'labels' #/ 'train'
        #target_path.mkdir(exist_ok=True)

        # Generate Exemplar lables
        if self.test_replay:
            from CL_replay import read_exemplar
            seen_ids = self.cl_states[cur_state]['knowing_class']['id']
            img_ids = read_exemplar()
            start_idx = 0
            gen_labels(target_path, img_ids, self.train_coco, seen_ids, start_idx)

        seen_ids = self.cl_states[cur_state]['new_class']['id']
        img_ids = self.train_coco.get_imgs_by_cats(seen_ids)

        if self.use_all_label:
            seen_ids = self.cl_states[cur_state]['knowing_class']['id']
            start_idx = 0
        else:
            start_idx = self.cl_states[cur_state]['num_past_class']
      
        gen_labels(target_path, img_ids, self.train_coco, seen_ids, start_idx)

        #Testing Labels
        target_path = DATA_ROOT / 'labels' #/ 'test'
        #target_path.mkdir(exist_ok=True)
        seen_ids = self.cl_states[cur_state]['knowing_class']['id']
        img_ids = self.test_coco.get_imgs_by_cats(seen_ids)
        start_idx = 0
        gen_labels(target_path, img_ids, self.test_coco, seen_ids, start_idx)