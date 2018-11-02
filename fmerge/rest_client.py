import requests


def upload_img():
    files = {'file': open('E://ocr//db//2.jpg', 'rb')}
    r = requests.post("http://127.0.0.1:9080/upload_img?s3path=/bucket0001/t1/1.jpg&num_in_folder=2",
                      files=files)
    print(r.text)


def send_result(url, json_dict):
    r = requests.post(url, json=json_dict)
    return r.text


if __name__ == '__main__':
    upload_img()
