import os
import zipfile
import argparse

def unzip(zip_file, extract_to):
    os.makedirs(extract_to, exist_ok = True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def zip_folder(folder_path, zip_file):
    with zipfile.ZipFile(zip_file, 'w') as ziph:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                ziph.write(os.path.join(root, file))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('from_path', help = 'Folder_path_OR_Zipfile_path')
    parser.add_argument('to_path',help = 'Folder_path_OR_Zipfile_path')
    parser.add_argument('--unzip','-un', action = 'store_true',help = 'Unzip_file')
    parser.add_argument('--zip','-z', action = 'store_true',help = 'Zip_folder')
    args = parser.parse_args()
    if args.unzip:
        unzip(args.from_path, args.to_path)
    elif args.zip:
        zip_folder(args.from_path, args.to_path)


if __name__ == '__main__':
    main()
    
