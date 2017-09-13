import sys
ID,FORM,LEMMA,UPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)
SID,SFORM,SLEMMA,SPLEMMA,SPOS,SPPOS,SFEAT,SPFEAT,SHEAD,SPHEAD,SDEPREL,SPDEPREL,PREDICATE,SENSE=range(14)
from collections import defaultdict

def read_conllu(f):
    comments=[]
    tokens=[]
    predicates={} # key: predicate_id, value: sense
    arguments=defaultdict(list) # key: argument_id (token[ID]), value: (predicate_id,role)
    for line in f:
        line=line.strip()
        if not line: # new sentence
            if tokens:
                yield comments,tokens,predicates,arguments
            comments,tokens,predicates,arguments=[],[],{},defaultdict(list)
        elif line.startswith("#"):
            comments.append(line)
        else: #normal line
            cols=line.split("\t")
            tokens.append(cols)
            # senses
            if "PBSENSE" in cols[MISC]:
                for m in cols[MISC].split("|"):
                    if m.startswith("PBSENSE"):
                        sense=m.split("=",1)[-1]
                        predicates[cols[ID]]=sense # ID (str) based indexing, not list based
            # arguments
            if "PBArg" in cols[DEPS]:
                for a in cols[DEPS].split("|"):
                    if "PBArg" in a:
                        idx,arg=a.split(":",1) # idx is predicate token[ID]
                        arguments[cols[ID]].append((idx,arg))
    else:
        if tokens:
            yield comments,tokens,predicates,arguments
  
def print_line(i,cols,predicates,predicate_counter,arguments):
    new_cols=["_"]*(14+len(predicates))
    for c,u in zip([ID,FORM,LEMMA,UPOS,FEAT,HEAD,DEPREL], [SID,SFORM,SLEMMA,SPOS,SFEAT,SHEAD,SDEPREL]):
        new_cols[u]=cols[c]
    if cols[ID] in predicates:
        new_cols[PREDICATE]="Y"
        new_cols[SENSE]=predicates[cols[ID]]
    if cols[ID] in arguments:
        for pidx,role in arguments[cols[ID]]:
            p_count=predicate_counter[pidx]
            new_cols[14+p_count]=role
    print("\t".join(new_cols)) 
            
            
def print_sdp(fname):

    f=open(fname, "rt", encoding="utf-8")
    for comments,tokens,predicates,arguments in read_conllu(f):
        predicate_counter={}
        for key, val in sorted(predicates.items(),key=lambda x: int(x[0])):
            predicate_counter[key]=len(predicate_counter)
        
        for i,token in enumerate(tokens):
            print_line(i,token,predicates,predicate_counter,arguments)
        print("")
    


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-f', '--file', type=str, required=True, help='File name')
 
    args = parser.parse_args()  
    
    print_sdp(args.file)   
    
