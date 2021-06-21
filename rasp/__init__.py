import rasp.core
import rasp.manual
import rasp.model


try:
  import rasp.daily
except ImportError:
  # download the latest utilities - github.com/yashbonde
  def get_gist(id):
    import requests, json, os
    d = json.loads(requests.get(f"https://api.github.com/gists/{id}").content.decode("utf-8"))
    for file_name in d["files"]: # save all the files exactly like on your gist
      here = os.path.split(os.path.abspath(__file__))[0]
      with open(os.path.join(here, file_name), "w") as f:
        f.write(d["files"][file_name]["content"])
  get_gist("62df9d16858a43775c22a6af00a8d707")
