# Copyright 2021 The PROST Authors
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
""" Template Formatting adapted from PROST [2], and originally Checklist [1].
  Sources:
    [1] https://github.com/marcotcr/checklist
    [2] https://github.com/nala-cub/prost
"""
import re
import string
from typing import Dict, TypeVar

import tree

TemplateObj = TypeVar('TemplateObj')


def add_article(noun: str) -> str:
  article = 'an' if noun[0].lower() in list('aeiou') else 'a'
  return '%s %s' % (article, noun)


def _remove_articles(k: str) -> str:
  """ Remove article prefixes [1]."""
  k = re.sub(r'\..*', '', k)
  k = re.sub(r'\[.*\]', '', k)
  # article
  k = re.sub(r'.*?:', '', k)
  return k


def _remove_counts(k: str) -> str:
  """ Remove count suffixes [1]."""
  return re.sub(r'\d+$', '', k)


def keys_to_var_names(keys):
  """ Convert list of keys to list of variable names. """
  # remove articles and clear duplicates
  var_keys = [_remove_counts(_remove_articles(k)) for k in keys]
  # the lists should be the same length
  assert len(keys) == len(var_keys)
  return var_keys


class SafeFormatter(string.Formatter):
  """ """

  def vformat(self, format_string, args, kwargs):
    args_len = len(args)  # for checking IndexError
    tokens = []
    for (lit, name, spec, conv) in self.parse(format_string):
      # re-escape braces that parse() unescaped
      lit = lit.replace('{', '{{').replace('}', '}}')
      # only lit is non-None at the end of the string
      if name is None:
        tokens.append(lit)
      else:
        # but conv and spec are None if unused
        conv = '!' + conv if conv else ''
        spec = ':' + spec if spec else ''
        # name includes indexing ([blah]) and attributes (.blah)
        # so get just the first part
        fp = name.split('[')[0].split('.')[0]
        # treat as normal if fp is empty (an implicit
        # positional arg), a digit (an explicit positional
        # arg) or if it is in kwargs
        if not fp or fp.isdigit() or fp in kwargs:
          tokens.extend([lit, '{', name, conv, spec, '}'])
        # otherwise escape the braces
        else:
          tokens.extend([lit, '{{', name, conv, spec, '}}'])
    format_string = ''.join(tokens)  # put the string back together
    # finally call the default formatter
    return string.Formatter.vformat(self, format_string, args, kwargs)


def recursive_format(obj: TemplateObj,
                     mapping: Dict,
                     ignore_missing: bool = False) -> TemplateObj:
  """Formats all strings within an object, using mapping

  Args:
    obj: Object (leaves must be strings, regardless of type)
    mapping: format dictionary, maps keys to values
    ignore_missing:  If True, will not throw exception if a string contains a
      tag not present in mapping, and will keep the tag instead.
  Returns:
    Object of the same type as obj, with strings formatted (tags replaced
    by their value)
  """

  def formatfn(x):
    fmt = SafeFormatter()
    formatz = (lambda x, m: x.format(**m)
               if not ignore_missing else fmt.format(x, **m))
    options = re.compile(r'{([^}]+):([^}]+)}')

    def mysub(match):
      options, thing = match.group(1, 2)
      ret = ''
      if 'a' in options:
        if ignore_missing and thing not in mapping:
          return match.group()
        else:
          word = formatz('{%s}' % thing, mapping)
          ret += '%s ' % add_article(word).split()[0]
      ret += '{%s}' % thing
      return ret

    x = options.sub(mysub, x)
    return formatz(x, mapping)

  return tree.map_structure(formatfn, obj)
