from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, jsonify, request
import os

fixedid = 'Fixed_ID.txt'
app = Flask('jky_c_server')
rtmpurl_pid_path = 'c_camera.txt'
basedir = os.path.abspath('.')
url_bianhao = 0
config_camera = 'c_camera.txt'
url_paths = 'url_pids.txt'


path = 'url_pids.txt'
result_path = '../results'
camera_ip_set = {}
ip_id = 0


@app.route('/server_v4', methods=['POST', 'GET'])
def api_upload():
    global url_bianhao, url_bianhao
    if request.method == 'GET':
        print("hello world")
        return "success"
    else:
        print('hello world')
        # try:
        '''
        rtmpurl = request.form['rtmpurl']
        applydate = request.form['applydate']


        camera_status = CAMERA_STATUS_START
        if "camera_status" in request.form:
            camera_status = request.form['camera_status']
            
            
               "BoxCode": "001",
             "CameraCode": "002",
             "ProjectID": "a79ec4e6-472d-413e-9585-66121b731cba"   
        '''
        status = request.json['status']
        # status = [{'001':'http://49.233.13.40:12077/WebAPI/api/aiReceiveData'},{'002':'http://49.233.13.40:12077/WebAPI/api/aiReceiveData'}]
        gcid = request.json['ProjectID']
        hzid = request.json['BoxCode']
        backurl = request.json['backurl']

        """
        0:qiyong
        -1:jinyong
        1:genghuan
        2:zengjia
        """
        if "ProjectID" in request.json:
            with open(fixedid, 'w', encoding='utf-8') as fix:
                fixid = gcid + ',' + hzid + ',' + backurl
                fix.writelines(fixid)
                fix.close()

            url_value = []
            urls = []

            for i in status:
                for val in i.keys():
                    camera_id = val
                    camera_url = i.get(val)
                    url = camera_url + '\n'
                    urls.append(url)
                    value = gcid + '_' + hzid + '_' + camera_id + '_' + camera_url + '\n'
                    url_value.append(value)

                print(url_value)
            with open(rtmpurl_pid_path, 'w', encoding='utf-8') as txt:
                txt.writelines(url_value)
                txt.close()

            with open(path, 'w', encoding='utf-8') as pa:
                pa.writelines(urls)
                pa.close()

            # time.sleep(2)

            return jsonify({"code": "1", "msg": "001"})

        else:
            return jsonify({"code": "0", "msg": "error"})
        #
        # return jsonify({"code": "1","msg":"523"})


if __name__ == '__main__':
    ap
    p.run(debug=True, host="localhost", port=8777)
