# Copyright 2021 Cory Paik. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Constants used during result processing."""
# Non-LaTex Names (base, and for Altair)
DF_ALT_COL_NAMES = {
    'name': 'Model',
    'model_type': 'Model',
    'model_size': 'Size',
    'object_group': 'Group',
    'spearman': 'Spearman Correlation',
    'pearson': 'Pearson Correlation',
    'kendalls_tau': 'Kendall\'s Tau',
    'acc': 'Top-1 Accuracy',
    'kl_div': 'KL Divergance',
    'dataset': 'Dataset',
    # datset splits
    'train': 'Train',
    'validation': 'Val',
    'test': 'Test',
    # deltas
    'delta_spearman': 'Delta Spearman Correlation',
    'delta_pearson': 'Delta Pearson Correlation',
    'delta_kendalls_tau': 'Delta Kendall\'s Tau',
    'corr_avg': 'Avg. Correlation',
    'delta_corr_avg': 'Avg. Delta Correlation',
    'jensenshannon_div': 'Jensen Shannon Divergence',
    'jensenshannon_dist': 'Jensen Shannon Distance'
}

DF_PRETTY_COL_NAMES = {
    **DF_ALT_COL_NAMES,
    'spearman': r'Spearman $\rho \uparrow$',
    'pearson': r'Pearson $r \uparrow$',
    'kendalls_tau': r"Kendall's $\tau \uparrow$",
    'acc': r'Acc@1 $\uparrow$',
    'kl_div': r'$\mathbf{D_{KL}} \downarrow$',
    'delta_spearman': r'$\Delta \rho \uparrow$',
    'delta_pearson': r'$\Delta r \uparrow$',
    'delta_kendalls_tau': r'$\Delta \tau \uparrow$',
    'corr_avg': r'Avg. Correlation $\uparrow$',
    'delta_corr_avg': r'Avg $\Delta$ Corr. $\uparrow$',
    'jensenshannon_dist': r'JS Dist. $\downarrow$',
    'jensenshannon_div': 'JSD $\\downarrow$',
}

NO_STD_METRICS = [
    'Freq',
    'delta_spearman',
    r'$\Delta \rho \uparrow$',
    'delta_pearson',
    r'$\Delta r \uparrow$',
    'delta_kendalls_tau',
    r'$\Delta \tau \uparrow$',
    r'Acc@1 $\uparrow$',
    'acc',
]
# DF_TEX_COL_NAMES = {
#   'name': 'Model',
#   'model_type': 'Model',
#   'object_group': 'Group',
#   'spearman': r'Spearman $\rho \uparrow$',
#   'pearson': r'Pearson $r \uparrow$',
#   'kendalls_tau': r"Kendall's $\tau \uparrow$",
#   'acc': r'Acc@1 $\uparrow$',
#   'kl_div': r'$\mathbf{D_{KL}} \downarrow$',
#   'dataset': 'Dataset'}

DF_REPLACE_MAP = {
    'model_type': {
        'albert': 'ALBERT',
        'bert': 'BERT',
        'roberta': 'RoBERTa',
        'openai': 'GPT',
        'gpt2': 'GPT-2',
        'vit': 'CLIP',
        'rn101': 'CLIP',
        'rn50': 'CLIP',
        'rn50x4': 'CLIP',
        'randinit': 'Random',
        'rawtokens': 'Raw Tokens'
    },
    'name': {
        'vit-b:32': 'CLIP ViT-B/32',
        'rn50': 'CLIP RN50',
        'rn101': 'CLIP RN101',
        'rn50x4': 'CLIP RN50x4',
        # gpt
        'openai-gpt': 'GPT',
        # gpt2
        'gpt2': 'GPT2',
        'gpt2-medium': 'GPT2 M',
        'gpt2-large': 'GPT2 L',
        'gpt2-xl': 'GPT2 XL',
        # bert
        'bert-base-uncased': 'BERT B',
        'bert-large-uncased': 'BERT L',
        # roberta
        'roberta-base': 'RoBERTa B',
        'roberta-large': 'RoBERTa L',
        # albert v1
        'albert-base-v1': 'ALBERT V1 B',
        'albert-large-v1': 'ALBERT V1 L',
        'albert-xlarge-v1': 'ALBERT V1 XL',
        'albert-xxlarge-v1': 'ALBERT V1 XXL',
        # albert v2
        'albert-base-v2': 'ALBERT V2 B',
        'albert-large-v2': 'ALBERT V2 L',
        'albert-xlarge-v2': 'ALBERT V2 XL',
        'albert-xxlarge-v2': 'ALBERT V2 XXL',
        # randomly initialized
        'rawtokens-bert': 'Raw Tokens',
        'randinit-bert': 'Random (BERT)',
        'randinit-clip': 'Random (CLIP)',
    },
    'object_group': {
        'single': 'Single',
        'multi': 'Multi',
        'any': 'Any'
    },
}

MODEL_ORDER = [
    # -- models
    'GPT',
    'GPT2',
    'BERT',
    'RoBERTa',
    'ALBERT',
    'ALBERT V1',
    'ALBERT V2',
    'CLIP',
    'Raw Tokens',
    'Random',
    # -- groups
    'Single',
    'Multi',
    'Any',
    'Total',
    # -- sizes
    '',
    'B',
    'M',
    'L',
    'XL',
    'XXL',
    'RN50',
    'RN101',
    'RN50x4',
    'ViT-B/32',
    # albert version sizes
    'V1 B',
    'V1 M',
    'V1 L',
    'V1 XL',
    'V1 XXL',
    'V2 B',
    'V2 M',
    'V2 L',
    'V2 XL',
    'V2 XXL',
    # -- rand
    '(BERT)',
    '(CLIP)',
    # -- datasets
    'Google Ngrams',
    'Wikipedia',
    'VQA',
    # -- average
    'Average',
]

COLORS = [
    'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple',
    'red', 'white', 'yellow'
]

METRIC_COLS = [
    'spearman',
    'pearson',
    'kendalls_tau',
    'acc',
    'kl_div',
    'jensenshannon_dist',
    'jensenshannon_div',
]
METRIC_PRECISION_MAP = {
    'spearman': 1,
    'kendalls_tau': 1,
    'acc': 1,
    'jensenshannon_dist': 2,
    'jensenshannon_div': 2,
    'Freq': 2,
}
