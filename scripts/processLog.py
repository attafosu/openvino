import sys
import os
import collections

def parseFile(fn=""):
    if len(fn)==0:
        return
    H = collections.defaultdict(set)
    with open(fn, "r") as fid:
        lines = fid.readlines()
        j = 0
        
        for line in lines:
            tup = line.split(": ")
            im_name = tup[0]
            #print("Im_name: {}".format(im_name))
            if not im_name.endswith("JPEG") or im_name.startswith("Image"):
                continue
            
            pred = line.split(": ")[-1].strip() # sample line: LSVRC2012_val_00000294.JPEG (Req 1013): 866 vs 866
            #print("Preds: {}".format(line.split(":")[-1].split("vs")[0].strip()))
            H[im_name].add(pred)
            j += 1
            if j ==50000:
                break
    
    for im_name in H:
        preds = H[im_name]
        if len(preds) > 1:
            print("{} has different predictions: {}".format(im_name, list(preds)))
        else:
            #pass
            print("{} has no duplicate predictions ({})".format(im_name, preds))


def main():
    if len(sys.argv) < 1:
        print("Requires log file")
        sys.exit(1)
    
    fname = sys.argv[1]
    print("Log file: {}".format(fname))
    if not os.path.exists(fname):
        print("Log file {} not found".format(fname))
        sys.exit(1)
        
    parseFile(fname)
    
    
if __name__=="__main__":
    main()