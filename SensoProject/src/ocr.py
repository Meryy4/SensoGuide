import easyocr

reader = easyocr.Reader(['en', 'fr'])

def extract_text_from_image(image_path):
    result = reader.readtext(image_path)
    #print("RAW OCR RESULT:", result) 

    text = " ".join([item[1] for item in result])
    
    return text

