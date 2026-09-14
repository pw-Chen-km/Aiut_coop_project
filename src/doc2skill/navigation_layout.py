"""Evidence-gated repairs for misleading captions and unpopulated part headings.

Operates on canonical source records without rewriting text or semantic grouping.
Ambiguous proposals are retained for review, never accepted merely by a keyword.
"""
from collections import Counter
from copy import deepcopy
import re
from statistics import median

from .comparison_source import _geometry
from .structure import _table_rows
from .source_sections import validate_parents

POLICY = 'navigation-layout-v1'


def _native(raw):
    return {x['self_ref']:x for name in ('texts','pictures','tables') for x in raw.get(name,[])}


def _g(item, raw):
    return _geometry({'prov':item.get('provenance',item.get('prov',[]))},raw)


def _below(picture, text):
    return (picture[0]==text[0] and 0<=text[2]-picture[4]<=max(24,2*(text[4]-text[2]))
            and abs((picture[1]+picture[3])/2-(text[1]+text[3])/2)<=max(12,(picture[3]-picture[1])*.08)
            and text[3]-text[1] < picture[3]-picture[1])


def _numbered_caption(text):
    return re.match(r'^\s*(\d+)\s+[A-Za-z]',text)


def _sync(fragment):
    sections=fragment['sections'];lookup={s['section_id']:s for s in sections}
    validate_parents(sections,{s['section_id']:s['parent_id'] for s in sections})
    for s in sections:
        if s['parent_id'] is not None:
            p=lookup[s['parent_id']];s.update(level=p['level']+1,path=p['path']+[s['title']])
        s['block_ids']=[b['block_id'] for b in fragment['blocks'] if b['section_id']==s['section_id']]
        pages=[p['page_index'] for b in fragment['blocks'] if b['section_id']==s['section_id']
               for p in b.get('provenance',[]) if p.get('page_index') is not None]
        s.update(page_start=min(pages) if pages else None,page_end=max(pages) if pages else None)
    for s in reversed(sections[1:]):
        p=lookup[s['parent_id']]
        for key,fn in [('page_start',min),('page_end',max)]:
            if s[key] is not None:p[key]=s[key] if p[key] is None else fn(p[key],s[key])
    owners={b['block_id']:b['section_id'] for b in fragment['blocks']}
    for unit in fragment['source_units']:unit['section_id']=owners[unit['block_id']]
    for fig in fragment['figures']:fig['section_id']=owners[fig['block_id']]
    assert Counter(owners.keys())==Counter(bid for s in sections for bid in s['block_ids'])


def repair_navigation_layout(fragment, raw):
    result=deepcopy(fragment);native=_native(raw)
    audit={'policy':POLICY,'caption_repairs':[],'part_repairs':[],'pending':[]}
    before_text=Counter((b['block_id'],b['text']) for b in result['blocks'])
    pictures=[x for x in raw.get('pictures',[]) if _g(x,raw)]
    captions=[x for x in raw.get('texts',[]) if x.get('label')=='caption' and _g(x,raw)]
    for s in list(result['sections'][1:]):
        if len(s['native_heading_refs'])!=1:continue
        ref=s['native_heading_refs'][0];item=native.get(ref);g=_g(item,raw) if item else None
        number=_numbered_caption(s['title'])
        if not g or not number:continue
        matches=[p for p in pictures if _below(_g(p,raw),g)]
        if not matches:continue
        # A native caption in the same numbered series provides independent
        # corroboration; uppercase or a number alone never suffices.
        peers=[c for c in captions if (m:=_numbered_caption(c['text']))
               and int(m[1])==int(number[1])+1 and _g(c,raw)[0]==g[0]
               and abs((_g(c,raw)[4]-_g(c,raw)[2])/(g[4]-g[2])-1)<=.15
               and any(_below(_g(p,raw),_g(c,raw)) for p in pictures)]
        if len(matches)!=1 or len(peers)!=1:
            audit['pending'].append({'kind':'caption','ref':ref,'reason':'Image adjacency lacks unique corroborating caption style/number sequence'})
            continue
        trial=deepcopy(result);removed=next(x for x in trial['sections'] if x['section_id']==s['section_id'])
        trial['sections'].remove(removed)
        for child in trial['sections']:
            if child['parent_id']==removed['section_id']:child['parent_id']=removed['parent_id']
        def printed_owner(block):
            bg=_g(block,raw)
            if not bg:raise ValueError('Missing source position for local ownership repair')
            candidates=[]
            for heading in trial['sections'][1:]:
                for href in heading['native_heading_refs']:
                    hg=_g(native[href],raw)
                    if hg and (hg[0]<bg[0] or hg[0]==bg[0] and hg[4]<=bg[2]):
                        candidates.append((hg[0],hg[4],heading['section_id']))
            return max(candidates)[2] if candidates else trial['sections'][0]['section_id']
        peer=peers[0]
        peer_picture=next(p for p in pictures if _below(_g(p,raw),_g(peer,raw)))
        linked_refs={ref,matches[0]['self_ref'],peer['self_ref'],peer_picture['self_ref']}
        moves=[]
        for b in trial['blocks']:
            if b['section_id']==removed['section_id'] or b['native_ref'] in linked_refs:
                old=b['section_id'];b['section_id']=printed_owner(b)
                if old!=b['section_id']:moves.append({'block_id':b['block_id'],'native_ref':b['native_ref'],'before':old,'after':b['section_id']})
            if b['native_ref']==ref:
                b.update(kind='caption',native_label='section_header',caption_repair_policy=POLICY)
        caption_block=next(b for b in trial['blocks'] if b['native_ref']==ref)
        for fig in trial['figures']:
            if fig['native_ref']==matches[0]['self_ref']:
                fig.setdefault('caption_block_ids',[])
                if caption_block['block_id'] not in fig['caption_block_ids']:fig['caption_block_ids'].append(caption_block['block_id'])
                fig.setdefault('caption_refs',[]).append({'$ref':ref})
                picture_block=next(b for b in trial['blocks'] if b['block_id']==fig['block_id'])
                picture_block.setdefault('caption_refs',[]).append({'$ref':ref})
        try:_sync(trial)
        except ValueError as exc:
            audit['pending'].append({'kind':'caption','ref':ref,'reason':str(exc)});continue
        result=trial
        audit['caption_repairs'].append({'removed_section_id':removed['section_id'],'ref':ref,'title':s['title'],
            'picture_ref':matches[0]['self_ref'],'corroborating_caption':peer['self_ref'],
            'evidence':{'heading_geometry':g,'picture_geometry':_g(matches[0],raw),'peer_geometry':_g(peer,raw)},
            'body_moves':moves,'reason':'Centered below a picture, matching the next native numbered caption in size and picture alignment; ownership reassessed by printed position'})
    # Part boundaries require a TOC plus a repeated typographic family of larger
    # uppercase part headings omitted from that TOC. No literal document titles.
    toc=[]
    for x in raw.get('tables',[]):
        if x.get('label')=='document_index':
            for row in _table_rows(x):
                cells=[c['docling_cell'] for c in row['cell_spans']]
                if cells:
                    toc.append({'title':' '.join(cells[0]['text'].split()),'table_ref':x['self_ref'],
                                'row':row['row_index'],'cell':cells[0]})
    toc_names=Counter(x['title'].casefold() for x in toc)
    sections=result['sections'];parts=[]
    for i,s in enumerate(sections[1:-1],1):
        if len(s['native_heading_refs'])!=1:continue
        x=native[s['native_heading_refs'][0]];g=_g(x,raw)
        nxt=sections[i+1];h=_g(native[nxt['native_heading_refs'][0]],raw)
        letters=''.join(c for c in s['title'] if c.isalpha())
        if (g and h and letters.isupper() and len(s['title'].split())>=2
                and 0<=h[0]-g[0]<=1 and abs(g[1]-h[1])<=20
                and g[4]-g[2]>(h[4]-h[2])*1.15
                and not toc_names[s['title'].casefold()] and toc_names[nxt['title'].casefold()]==1):
            parts.append((i,s,g,nxt))
    if parts:
        height=median(g[4]-g[2] for _,_,g,_ in parts);left=median(g[1] for _,_,g,_ in parts)
        supported=(len(parts)>=2 and len(toc)>=3 and all(abs((g[4]-g[2])/height-1)<=.15 and abs(g[1]-left)<=10 for _,_,g,_ in parts))
        if not supported:
            audit['pending'].extend({'kind':'part','ref':s['native_heading_refs'][0],
                'reason':'Insufficient repeated part style and unique TOC evidence'} for _,s,_,_ in parts)
        else:
            trial=deepcopy(result);by={s['section_id']:s for s in trial['sections']};changes=[]
            for j,(i,s,g,nxt) in enumerate(parts):
                end=parts[j+1][0] if j+1<len(parts) else len(sections)
                interval={x['section_id'] for x in sections[i:end]};part=by[s['section_id']]
                if part['parent_id']!=sections[0]['section_id']:
                    changes.append({'id':part['section_id'],'before':part['parent_id'],'after':sections[0]['section_id']})
                    part['parent_id']=sections[0]['section_id']
                children=[]
                for original in sections[i+1:end]:
                    child=by[original['section_id']]
                    if child['parent_id'] not in interval:
                        # Only attach TOC-confirmed headings. Unknown external
                        # owners would make the interval unsafe to auto-repair.
                        if toc_names[child['title'].casefold()]!=1:
                            audit['pending'].append({'kind':'part','ref':s['native_heading_refs'][0],
                                'reason':'Unconfirmed child within candidate interval','title':child['title']})
                            supported=False;break
                        changes.append({'id':child['section_id'],'before':child['parent_id'],'after':part['section_id']})
                        child['parent_id']=part['section_id'];children.append(child['title'])
                if not supported:break
            if supported:
                try:_sync(trial)
                except ValueError as exc:audit['pending'].append({'kind':'part','reason':str(exc)})
                else:
                    result=trial
                    audit['part_repairs']=[{'part_titles':[s['title'] for _,s,_,_ in parts],
                        'part_refs':[s['native_heading_refs'][0] for _,s,_,_ in parts],
                        'toc_evidence':toc,'changes':changes,
                        'reason':'Repeated larger uppercase part style; following TOC-confirmed topics bounded by next part; internal parents preserved'}]
    _sync(result)
    assert before_text==Counter((b['block_id'],b['text']) for b in result['blocks'])
    assert fragment['furniture']==result['furniture']
    audit['text_preserved']=True
    audit['status']='needs_review' if audit['pending'] else 'complete'
    result['audit']['navigation_layout']=audit
    return result,audit
