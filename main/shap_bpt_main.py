from utility.patch_shap_bpt import single_pipeline_bpt

if __name__ == "__main__":
    
    in_img_path: str = "C:\\Users\\biagi\\OneDrive\\Documents\\Desktop\\progetti\\sota_maker\\dataset\\test\\Byzantine\\Frag_2.png"
    out_json_path: str = "out.json"
    
    single_pipeline_bpt(in_img_path=in_img_path, out_json_path=out_json_path)