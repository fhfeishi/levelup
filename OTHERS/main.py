from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
import uvicorn


from gen_doc import do_generate_word, do_generate_pdf, do_generate_xls
import sys
from measurelib import measure
from measurelib import measure2
#import uvicorn
from cal_circule import calurate_circle_diameter

app = FastAPI() 
# app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/measure_atom")
async def measure_atom( reqbody: dict ):
    source_path = reqbody["source_path"]
    out_path = reqbody["out_path"]
    out_path2 = reqbody["out_path2"]

    model_type = reqbody["model_type"]
    model_name = reqbody["model_name"]
    if model_name=='单层环形':
        print(reqbody)
        errorcode,maxt_all,ave_all,condentricity_all,\
        eccentricity_all,t_six,mint_six,maxt_six,\
        ave_six,d_x,d_y,d_xy,ellipticity,\
        condentricity_six,eccentricity_six,area = measure.measure(source_path, out_path)
        res = {}
        res["errorcode"] = errorcode
        res["source_path"] = source_path
        res["out_path"] = out_path
        res['t_six'] = t_six
        res['d_x'] = d_x
        res['d_y'] = d_y
        res['d_xy'] = d_xy
        res['mint_six'] = mint_six
        res['ave_six'] = ave_six
        res['ave_all'] = ave_all
        res['area'] = area
        res['maxt_six'] = maxt_six
        res['maxt_all'] = maxt_all
        res['condentricity_all'] = condentricity_all
        res['eccentricity_all'] = eccentricity_all
        res['ellipticity'] = ellipticity
        res['condentricity_six'] = condentricity_six
        res['eccentricity_six'] = eccentricity_six
        print("response ", res )
        return res

    if model_name=='双层环形':
        print(reqbody)
        errorcode,\
maxt_all1,ave_all1,condentricity_all1,eccentricity_all1,\
t_six1,mint_six1,maxt_six1,ave_six1,d_x1,d_y1,d_xy1,ellipticity1,condentricity_six1,eccentricity_six1,area1,\
maxt_all2,ave_all2,condentricity_all2,eccentricity_all2,\
t_six2,mint_six2,maxt_six2,ave_six2,d_x2,d_y2,d_xy2,ellipticity2,condentricity_six2,eccentricity_six2,area2,\
maxt_all3,ave_all3,condentricity_all3,eccentricity_all3,\
t_six3,mint_six3,maxt_six3,ave_six3,d_x3,d_y3,d_xy3,ellipticity3,condentricity_six3,eccentricity_six3,area3\
=measure2.measure(source_path,out_path, out_path2)
        res = {}
        res["errorcode"] = errorcode
        res["source_path"] = source_path
        res["out_path"] = out_path
        res["out_path2"] = out_path2

        res['t_six'] = t_six1
        res['d_x'] = d_x1
        res['d_y'] = d_y1
        res['d_xy'] = d_xy1
        res['mint_six'] = mint_six1
        res['ave_six'] = ave_six1
        res['ave_all'] = ave_all1
        res['area'] = area1
        res['maxt_six'] = maxt_six1
        res['maxt_all'] = maxt_all1
        res['condentricity_all'] = condentricity_all1
        res['eccentricity_all'] = eccentricity_all1
        res['ellipticity'] = ellipticity1
        res['condentricity_six'] = condentricity_six1
        res['eccentricity_six'] = eccentricity_six1

        res['t_six2'] = t_six2
        res['d_x2'] = d_x2
        res['d_y2'] = d_y2
        res['d_xy2'] = d_xy2
        res['mint_six2'] = mint_six2
        res['ave_six2'] = ave_six2
        res['ave_all2'] = ave_all2
        res['area2'] = area2
        res['maxt_six2'] = maxt_six2
        res['maxt_all2'] = maxt_all2
        res['condentricity_all2'] = condentricity_all2
        res['eccentricity_all2'] = eccentricity_all2
        res['ellipticity2'] = ellipticity2
        res['condentricity_six2'] = condentricity_six2
        res['eccentricity_six2'] = eccentricity_six2

        # 333333
        res['t_six3'] = t_six3
        res['d_x3'] = d_x3
        res['d_y3'] = d_y3
        res['d_xy3'] = d_xy3
        res['mint_six3'] = mint_six3
        res['ave_six3'] = ave_six3
        res['ave_all3'] = ave_all3
        res['area3'] = area3
        res['maxt_six3'] = maxt_six3
        res['maxt_all3'] = maxt_all3
        res['condentricity_all3'] = condentricity_all3
        res['eccentricity_all3'] = eccentricity_all3
        res['ellipticity3'] = ellipticity3
        res['condentricity_six3'] = condentricity_six3
        res['eccentricity_six3'] = eccentricity_six3
        print("response ", res )
        return res

@app.post("/generate_doc")
async def generate_doc( reqbody: dict ):
    source_path = reqbody["source_path"]
    image_path = reqbody["image_path"]
    data_maps = reqbody["data_maps"]
    target_types = reqbody["target_types"]
    print(reqbody)
    print("source_path ", source_path, " data_maps ", data_maps)
    
    word_path = do_generate_word(source_path, data_maps, image_path)
    pdf_path = do_generate_pdf(word_path)
    xls_path = do_generate_xls(source_path, data_maps)

    res = {}
    if "doc" in target_types:
        res['doc'] = word_path
    if "pdf" in target_types:
        res['pdf'] = pdf_path
    if "xls" in target_types:
        res['xls'] = xls_path

    print("response ", res )
    return res

@app.post("/cal_diameter")
async def cal_diameter( reqbody: dict ):
    source_path = reqbody["source_path"]
    print(reqbody)
    print("source_path ", source_path)

    res = {}
    num, d = calurate_circle_diameter(source_path,True)
    res['diameter'] = str(d)
    res['hasOneCircle'] = str(num)
    print("response ", res )
    return res

@app.post("/test_api")
async def test_api( reqbody : dict):
    print(reqbody)
    return {"abc":123} 



if __name__ == "__main__":
    uvicorn.run("main:app")#, host="locolhost", port=8000)

