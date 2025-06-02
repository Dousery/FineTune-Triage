import tarfile
import os

def extract_tar():
    tar_file = "merged_medical_model.tar.gz"
    
    if not os.path.exists(tar_file):
        print(f" Hata: {tar_file} dosyası bulunamadı!")
        return
    
    print(f" {tar_file} dosyası açılıyor...")
    
    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            print("\n Arşiv içeriği:")
            for member in tar.getmembers():
                print(f"- {member.name}")
            
            # Dosyaları çıkar
            tar.extractall('.')
            
        print("\n Dosyalar başarıyla çıkarıldı!")
        print(" Çıkarılan dosyalar 'merged_model' klasöründe bulunabilir.")
        
    except Exception as e:
        print(f" Hata: {str(e)}")

if __name__ == "__main__":
    extract_tar() 