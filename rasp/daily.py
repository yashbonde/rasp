# whole bunch of utility functions I use in day to day
# - @yashbonde / https://github.com/yashbonde
#
# Now this is a simple script and cannot be loaded like a package
# so you'll need to import it. This is how you can do it
"""
try:
  from daily import *
except ImportError as e:
  import requests
  x = requests.get(<url_to_raw>).content
  with open("daily.py", "wb") as f:
    f.write(x)
  from daily import *
"""

import logging
logging.basicConfig(level="INFO")
def log(x: str, *args):
  x = str(x)
  for y in args:
    x += " " + str(y)
  logging.info(x)


def fetch(url):
  # efficient loading of URLS
  import os, tempfile, hashlib, requests
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp) and os.stat(fp).st_size > 0:
    with open(fp, "rb") as f:
        dat = f.read()
  else:
    print("fetching", url)
    dat = requests.get(url).content
    with open(fp+".tmp", "wb") as f:
      f.write(dat)
    os.rename(fp+".tmp", fp)
  return dat


def get_files_in_folder(folder, ext = [".txt"]):
  # this method is faster than glob
  import os
  all_paths = []
  for root,_,files in os.walk(folder):
    for f in files:
      for e in ext:
        if f.endswith(e):
          all_paths.append(os.path.join(root,f))
  return all_paths


def json_load(path):
  # load any JSON like file with comment strings '//'
  # json files should have description strings so it's more friendly
  # but it won't load with json.load(f) so read the file, remove the
  # comments and json.loads(text)
  import json, re
  with open(path, 'r') as f:
    text = f.read()
  text = re.sub(r"\s*(\/{2}.*)\n", "\n", text)
  config = json.loads(text)
  return config


def folder(x):
  # get the folder of this file path
  import os
  return os.path.split(os.path.abspath(x))[0]


# class to handle hashing methods
class Hashlib:
  # local imports don't work!
  # import hashlib
  
  def sha256(x):
    import hashlib
    x = x if isinstance(x, bytes) else x.encode("utf-8")
    return hashlib.sha256(x).hexdigest()
  
  def md5(x):
    import hashlib
    x = x if isinstance(x, bytes) else x.encode("utf-8")
    return hashlib.md5(x).hexdigest()
