# sdp here means the conll09 format
import sys
ID,FORM,LEMMA,UPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)
SID,SFORM,SLEMMA,SPLEMMA,SPOS,SPPOS,SFEAT,SPFEAT,SHEAD,SPHEAD,SDEPREL,SPDEPREL,PREDICATE,SENSE,ARG=range(15)
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
            
def read_sdp(f):

    comments=[]
    tokens=[]
    predicates={} # key: predicate_id, value: sense
    arguments=defaultdict(list) # key: argument_id (token[ID]), value: (predicate_id,role)
    tmp_preds=[]
    for line in f:
        line=line.strip()
        if not line: # empty line
            continue
        if line.startswith("1\t") and len(tokens)>0: # new sentence, yield current
            # turn predicated counters into real token IDs
            new_arguments={}
            for key in arguments:
                new_arguments[key]=[]
                for pred_c,role in arguments[key]:
                    new_arguments[key].append((tmp_preds[pred_c],role))
            #
            yield comments,tokens,predicates,new_arguments
            comments,tokens,predicates,arguments,tmp_preds=[],[],{},defaultdict(list),[]
        cols=line.split("\t")
        tokens.append(cols)
        # senses
        if "Y" in cols[PREDICATE]: # this is predicate
            tmp_preds.append(cols[ID])
            sense=cols[SENSE]
            predicates[cols[ID]]=sense # ID (str) based indexing, not list based
        # arguments
        for i in range(ARG,len(cols)):
            if cols[i]!="_": # this is argument
                pred_counter=i-ARG # 0-based counter of predicates
                arguments[cols[ID]].append((pred_counter,cols[i])) # key is the ID of the argument word (str), value is the counter of the predicate (int) and the role 
    else:
        if tokens:
            # turn predicated counters into real token IDs
            new_arguments={}
            for key in arguments:
                new_arguments[key]=[]
                for pred_c,role in arguments[key]:
                    
                    new_arguments[key].append((tmp_preds[pred_c],role))
            #
            yield comments,tokens,predicates,new_arguments
  
def print_sdp_line(i,cols,predicates,predicate_counter,arguments):
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
            print_sdp_line(i,token,predicates,predicate_counter,arguments)
        print("")
        
def print_conllu_line(i,cols,predicates,arguments):
    # TODO: do not eat deps
    new_cols=["_"]*10
    for c,u in zip([ID,FORM,LEMMA,UPOS,FEAT,HEAD,DEPREL], [SID,SFORM,SLEMMA,SPOS,SFEAT,SHEAD,SDEPREL]):
        new_cols[c]=cols[u]
    if cols[SID] in predicates:
        new_cols[MISC]="PBSENSE="+predicates[cols[SID]] # sense
    if cols[SID] in arguments:
        args=[h+":PBArg_"+r for h,r in arguments[cols[SID]]]
        new_cols[DEPS]="|".join(sorted(args))
    print("\t".join(new_cols))
    
def print_conllu(fname):

    f=open(fname, "rt", encoding="utf-8")
    for comments,tokens,predicates,arguments in read_sdp(f):
        for i,token in enumerate(tokens):
            print_conllu_line(i,token,predicates,arguments)
        print("")

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-f', '--file', type=str, required=True, help='File name')
    g.add_argument('-r', '--reversed', action='store_true', default=False, help='Reverse the process --> read sdp and print conllu')
 
    args = parser.parse_args()  
    
    if args.reversed:
        print("Reading sdp and printing conllu",file=sys.stderr)
        print_conllu(args.file)
    else:
        print("Reading conllu and printing sdp",file=sys.stderr)
        print_sdp(args.file)   
    
