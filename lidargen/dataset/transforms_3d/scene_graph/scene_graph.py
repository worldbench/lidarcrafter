import os
import clip
import pickle
import torch
import numpy as np
import copy

changed_relationships_dict = {
        'left': 'right',
        'right': 'left',
        'front': 'behind',
        'behind': 'front',
        'bigger than': 'smaller than',
        'smaller than': 'bigger than',
        'taller than': 'shorter than',
        'shorter than': 'taller than',
        'close by': 'close by',
    }

class SceneGraphAssigner:
    def __init__(self, output_path, relationship_file_path, classnames_file_path, split):
        self.output_path = output_path
        self.split = split
        self.with_CLIP = True
        self.vocab = {}
        self.catfile = classnames_file_path
        self.cat = {}
        self.with_changes = True
        self.eval = False if self.split == 'train' else True
        self.eval_type = 'none'
        with open(self.catfile, "r") as f:
            self.vocab['object_idx_to_name'] = f.readlines()
        with open(relationship_file_path, "r") as f:
            self.vocab['pred_idx_to_name'] = ['in\n']
            self.vocab['pred_idx_to_name']+=f.readlines()

        self.relationships = self.read_relationships(relationship_file_path)
        self.relationships_dict = dict(zip(self.relationships,range(len(self.relationships))))
        self.relationships_dict_r = dict(zip(self.relationships_dict.values(), self.relationships_dict.keys()))

        with open(self.catfile, 'r') as f:
            for line in f:
                category = line.rstrip()
                self.cat[category] = category

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))

        self.build_clip()

    def build_clip(self):
        self.cond_model, preprocess = clip.load("ViT-B/32", device='cuda')
        self.cond_model_cpu, preprocess_cpu = clip.load("ViT-B/32", device='cpu')

    def get_unique_names(self, obj_json):
        unique_obj_json = []
        objs_count = {}

        for obj in obj_json:
            if obj_json.count(obj) > 1:
                count = objs_count.get(obj, 0) + 1
                objs_count[obj] = count
                new_obj = f"{obj}{count}"
                unique_obj_json.append(new_obj)
            else:
                unique_obj_json.append(obj)
        return unique_obj_json

    def get_words(self, gt_names, gt_box_relationships):
        words = []
        rel_json = gt_box_relationships
        obj_json = list(gt_names)

        objs_count = {}
        unique_obj_json = []

        for obj in obj_json:
            if obj_json.count(obj) > 1:
                count = objs_count.get(obj, 0) + 1
                objs_count[obj] = count
                new_obj = f"{obj}{count}"
                unique_obj_json.append(new_obj)
            else:
                unique_obj_json.append(obj)

        for r in rel_json:
            words.append(unique_obj_json[r[0]] + ' ' + self.relationships[r[1]] + ' ' + unique_obj_json[r[2]]) # TODO check
        return words

    def assign_item(self, idx, data_dict):
        scan_id = str(idx).zfill(7)
        if self.with_CLIP:
            self.clip_feats_path = os.path.join(self.output_path, self.split,
                                                'CLIP_{}.pkl'.format(scan_id))
            os.makedirs(os.path.join(self.output_path, self.split), exist_ok=True)

        boxes = data_dict['scaled_gt_boxes'][:,:8]
        # box with traj 'gt_fut_trajs
        gt_mask = np.ones([boxes.shape[0], 8+12], dtype=np.bool_)
        gt_mask[0,:8] = False
        gt_fut_trajs = data_dict['gt_fut_trajs'].reshape(boxes.shape[0], -1)
        boxes = np.concatenate([boxes, gt_fut_trajs], axis=1)
        gt_fut_masks = data_dict['gt_fut_masks']
        gt_fut_masks = np.repeat(gt_fut_masks[:, :, np.newaxis], 2, axis=2).astype(np.float32)
        gt_mask[:, 8:] = gt_fut_masks.reshape(boxes.shape[0], -1)
        boxes = np.concatenate([boxes, gt_mask], axis=1)

        triples = []
        words = []
        rel_json = data_dict['gt_box_relationships']
        obj_json = list(data_dict['gt_names'])
        state_json = list(data_dict['gt_fut_states'])

        objs_count = {}
        unique_obj_json = []

        for obj in obj_json:
            if obj_json.count(obj) > 1:
                count = objs_count.get(obj, 0) + 1
                objs_count[obj] = count
                new_obj = f"{obj}{count}"
                unique_obj_json.append(new_obj)
            else:
                unique_obj_json.append(obj)

        for r in rel_json:
            triples.append(r)
            words.append(unique_obj_json[r[0]] + ' ' + self.relationships[r[1]] + ' ' + unique_obj_json[r[2]]) # TODO check

        obj_and_state_words = []
        for object_id, object_name in enumerate(obj_json):
            obj_and_state_words.append(object_name + ' ' + 'will' + ' ' + state_json[object_id].lower())

        if self.with_CLIP:
            # If precomputed features exist, we simply load them
            if os.path.exists(self.clip_feats_path):
                clip_feats_dic = pickle.load(open(self.clip_feats_path, 'rb'))
                clip_feats_ins = clip_feats_dic['instance_feats']
                clip_feats_rel = clip_feats_dic['rel_feats']
                if not isinstance(clip_feats_ins, list):
                    clip_feats_ins = list(clip_feats_ins)

        output = {}
        # if features are requested but the files don't exist, we run all loaded cats and triples through clip
        # to compute them and then save them for future usage
        if self.with_CLIP and (not os.path.exists(self.clip_feats_path) or clip_feats_ins is None) and self.cond_model is not None:
            feats_rel = {}
            with torch.no_grad():

                text_obj = clip.tokenize(obj_and_state_words).to('cuda')
                feats_ins = self.cond_model.encode_text(text_obj).detach().cpu().numpy()
                text_rel = clip.tokenize(words).to('cuda')
                rel = self.cond_model.encode_text(text_rel).detach().cpu().numpy()
                for i in range(len(words)):
                    feats_rel[words[i]] = rel[i]

            clip_feats_in = {}
            clip_feats_in['instance_feats'] = feats_ins
            clip_feats_in['rel_feats'] = feats_rel
            path = os.path.join(self.clip_feats_path)

            pickle.dump(clip_feats_in, open(path, 'wb'))
            clip_feats_ins = list(clip_feats_in['instance_feats'])
            clip_feats_rel = clip_feats_in['rel_feats']


        # prepare outputs
        output['encoder'] = {}
        output['encoder']['objs'] = [self.classes[obj] for obj in obj_json]
        output['encoder']['triples'] = triples
        output['encoder']['boxes'] = list(boxes)
        output['encoder']['words'] = words
        output['encoder']['unique_objs'] = unique_obj_json

        if self.with_CLIP:
            output['encoder']['text_feats'] = clip_feats_ins
            clip_feats_rel_new = []
            if clip_feats_rel != None:
                for word in words:
                    clip_feats_rel_new.append(clip_feats_rel[word])
                output['encoder']['rel_feats'] = clip_feats_rel_new

        output['manipulate'] = {}
        if not self.with_changes:
            output['manipulate']['type'] = 'none'
            output['decoder'] = copy.deepcopy(output['encoder'])
        else:
            if not self.eval:
                if self.with_changes:
                    output['manipulate']['type'] = ['relationship', 'addition', 'none'][
                        # 1]
                        np.random.randint(3)]  # removal is trivial - so only addition and rel change
                else:
                    output['manipulate']['type'] = 'none'
                output['decoder'] = copy.deepcopy(output['encoder'])
                if len(output['encoder']['objs']) <=2:
                    output['manipulate']['type'] = 'none'
                if output['manipulate']['type'] == 'addition':
                    node_id, node_removed, node_clip_removed, triples_removed, triples_clip_removed, words_removed = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added_node_id'] = node_id
                        output['manipulate']['added_node_class'] = node_removed
                        output['manipulate']['added_node_clip'] = node_clip_removed
                        output['manipulate']['added_triples'] = triples_removed
                        output['manipulate']['added_triples_clip'] = triples_clip_removed
                        output['manipulate']['added_words'] = words_removed
                    else:
                        output['manipulate']['type'] = 'none'
                elif output['manipulate']['type'] == 'relationship':
                    rel, original_triple, suc = self.modify_relship(output['encoder']) # why modify encoder side? Because the changed edge doesn't need to make sense. I need to make sure that the one from the decoder side makes sense
                    if suc:
                        output['manipulate']['original_relship'] = (rel, original_triple)
                    else:
                        output['manipulate']['type'] = 'none'
            else:
                output['manipulate']['type'] = self.eval_type
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id, node_removed, node_clip_removed, triples_removed, triples_clip_removed, words_removed = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added_node_id'] = node_id
                        output['manipulate']['added_node_class'] = node_removed
                        output['manipulate']['added_node_clip'] = node_clip_removed
                        output['manipulate']['added_triples'] = triples_removed
                        output['manipulate']['added_triples_clip'] = triples_clip_removed
                        output['manipulate']['added_words'] = words_removed
                    else:
                        return -1
                elif output['manipulate']['type'] == 'relationship':
                    # this should be modified from the decoder side, because during test we have to evaluate the real setting, and can make the change have the real meaning.
                    rel, original_triple, suc = self.modify_relship(output['decoder'], interpretable=True)
                    if suc:
                        output['manipulate']['original_relship'] = (rel, original_triple)
                    else:
                        return -1

        # torchify
        output['encoder']['objs'] = torch.from_numpy(np.array(output['encoder']['objs'], dtype=np.int64)) # this is changed
        output['encoder']['triples'] = torch.from_numpy(np.array(output['encoder']['triples'], dtype=np.int64))
        output['encoder']['boxes'] = torch.from_numpy(np.array(output['encoder']['boxes'], dtype=np.float32))
        if self.with_CLIP:
            output['encoder']['text_feats'] = torch.from_numpy(np.array(output['encoder']['text_feats'], dtype=np.float32)) # this is changed
            output['encoder']['rel_feats'] = torch.from_numpy(np.array(output['encoder']['rel_feats'], dtype=np.float32))

        # these two should have the same amount.
        output['decoder']['objs'] = torch.from_numpy(np.array(output['decoder']['objs'], dtype=np.int64))

        output['decoder']['triples'] = torch.from_numpy(np.array(output['decoder']['triples'], dtype=np.int64)) # this is changed
        output['decoder']['boxes'] = torch.from_numpy(np.array(output['decoder']['boxes'], dtype=np.float32))
        if self.with_CLIP:
            output['decoder']['text_feats'] = torch.from_numpy(np.array(output['decoder']['text_feats'], dtype=np.float32))
            output['decoder']['rel_feats'] = torch.from_numpy(np.array(output['decoder']['rel_feats'], dtype=np.float32)) # this is changed

        output['scan_id'] = scan_id

        return output

    def remove_node_and_relationship(self, graph):
        """ Automatic random removal of certain nodes at training time to enable training with changes. In that case
        also the connecting relationships of that node are removed

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :return: index of the removed node
        """

        node_id = -1
        # dont remove layout components, like floor. those are essential
        excluded = [self.classes['ego']]

        trials = 0
        while node_id < 0 or graph['objs'][node_id] in excluded:
            if trials > 100:
                return -1
            trials += 1
            node_id = np.random.randint(len(graph['objs']) - 1)

        node_removed = graph['objs'].pop(node_id)
        node_clip_removed = None
        if self.with_CLIP:
            node_clip_removed = graph['text_feats'].pop(node_id)
        try:
            graph['boxes'].pop(node_id)
        except:
            breakpoint()

        to_rm = []
        triples_clip_removed_list = []
        words_removed_list = []
        for i,x in reversed(list(enumerate(graph['triples']))):
            sub, pred, obj = x
            if sub == node_id or obj == node_id:
                to_rm.append(x)
                if self.with_CLIP:
                    triples_clip_removed_list.append(graph['rel_feats'].pop(i))
                    words_removed_list.append(graph['words'].pop(i))

        triples_removed = copy.deepcopy(to_rm)
        while len(to_rm) > 0:
            graph['triples'].remove(to_rm.pop(0))

        for i in range(len(graph['triples'])):
            if graph['triples'][i][0] > node_id:
                graph['triples'][i][0] -= 1

            if graph['triples'][i][2] > node_id:
                graph['triples'][i][2] -= 1

        # node_id: instance_id; node_removed: class_id; triples_removed: relations (sub_id, edge_id, obj_id)
        return node_id, node_removed, node_clip_removed, triples_removed, triples_clip_removed_list, words_removed_list

    def modify_relship(self, graph, interpretable=False):
        """ Change a relationship type in a graph

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :param interpretable: boolean, if true choose a subset of easy to interpret relations for the changes
        :return: index of changed triplet, a tuple of affected subject & object, and a boolean indicating if change happened
        """

        # rels 26 -> 0
        '''15 same material as' '14 same super category as' '13 same style as' '12 symmetrical to' '11 shorter than' '10 taller than' '9 smaller than'
         '8 bigger than' '7 standing on' '6 above' '5 close by' '4 behind' '3 front' '2 right' '1 left'
         '0: none'''
        # subset of edge labels that are spatially interpretable (evaluatable via geometric contraints)
        interpretable_rels = [0, 1, 2, 3, 5, 6, 7, 8]
        did_change = False
        trials = 0
        excluded = []
        eval_excluded = []

        while not did_change and trials < 1000:
            idx = np.random.randint(len(graph['triples']))
            sub, pred, obj = graph['triples'][idx]
            trials += 1

            if graph['objs'][obj] in excluded or graph['objs'][sub] in excluded:
                continue
            if interpretable:
                if graph['objs'][obj] in eval_excluded or graph['objs'][sub] in eval_excluded: # don't use the floor
                    continue
                if pred not in interpretable_rels:
                    continue
                else:
                    new_pred = self.relationships_dict[changed_relationships_dict[self.relationships_dict_r[pred]]]
            else:
                new_pred = np.random.randint(0, 9)
                if new_pred == pred:
                    continue

            graph['words'][idx] = graph['words'][idx].replace(self.relationships_dict_r[graph['triples'][idx][1]],self.relationships_dict_r[new_pred])
            graph['changed_id'] = idx

            # When interpretable is false, we can make things from the encoder side not existed, so that make sure decoder side is the real data.
            # When interpretable is true, we can make things from the decoder side not existed, so that make sure what we test (encoder side) is the real data.
            graph['triples'][idx][1] = new_pred # this new_pred may even not exist.

            did_change = True

        # idx: idx-th triple; (sub, pred, obj): previous triple
        return idx, (sub, pred, obj), did_change

    def read_relationships(self, read_file):
        """load list of relationship labels

        :param read_file: path of relationship list txt file
        """
        relationships = []
        with open(read_file, 'r') as f:
            for line in f:
                relationship = line.rstrip().lower()
                relationships.append(relationship)
        return relationships