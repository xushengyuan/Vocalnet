import xml.dom.minidom
import random
dictp={
'-': 0 ,
'4': 0.003921569 ,
'7': 0.007843137 ,
'@_n': 0.011764706 ,
'@`': 0.015686275 ,
'@N': 0.019607843 ,
'@r': 0.023529412 ,
'@U': 0.02745098 ,
'{': 0.031372549 ,
'4\'': 0.035294118 ,
'a': 0.039215686 ,
'a_n': 0.043137255 ,
'ai': 0.047058824 ,
'aI': 0.050980392 ,
'AN': 0.054901961 ,
'AU': 0.058823529 ,
'b': 0.062745098 ,
'b\'': 0.066666667 ,
'bh': 0.070588235 ,
'C': 0.074509804 ,
'd': 0.078431373 ,
'd\'': 0.082352941 ,
'dh': 0.08627451 ,
'dZ': 0.090196078 ,
'e': 0.094117647 ,
'e@': 0.098039216 ,
'ei': 0.101960784 ,
'f': 0.105882353 ,
'g': 0.109803922 ,
'g\'': 0.11372549 ,
'gh': 0.117647059 ,
'h': 0.121568627 ,
'h\\': 0.125490196 ,
'i': 0.129411765 ,
'i:': 0.133333333 ,
'i@U': 0.137254902 ,
'i\\': 0.141176471 ,
'i_n': 0.145098039 ,
'i`': 0.149019608 ,
'ia': 0.152941176 ,
'iAN': 0.156862745 ,
'iAU': 0.160784314 ,
'iE_n': 0.164705882 ,
'iE_r': 0.168627451 ,
'iN': 0.17254902 ,
'iUN': 0.176470588 ,
'j': 0.180392157 ,
'k': 0.184313725 ,
'k\'': 0.188235294 ,
'k_h': 0.192156863 ,
'kh': 0.196078431 ,
'l': 0.2 ,
'l@': 0.203921569 ,
'l0': 0.207843137 ,
'm': 0.211764706 ,
'm\'': 0.215686275 ,
'n': 0.219607843 ,
'N\'': 0.223529412 ,
'N\\': 0.22745098 ,
'o': 0.231372549 ,
'O:': 0.235294118 ,
'O@': 0.239215686 ,
'OI': 0.243137255 ,
'p': 0.247058824 ,
'p\'': 0.250980392 ,
'p\\': 0.254901961 ,
'p\\\'': 0.258823529 ,
'p_h': 0.262745098 ,
'ph': 0.266666667 ,
'Q': 0.270588235 ,
'Q@': 0.274509804 ,
'r': 0.278431373 ,
's': 0.282352941 ,
's\\': 0.28627451 ,
's`': 0.290196078 ,
't': 0.294117647 ,
't\'': 0.298039216 ,
't_h': 0.301960784 ,
'ta': 0.305882353 ,
'ts_h': 0.309803922 ,
'th': 0.31372549 ,
'ts': 0.317647059 ,
'ts\\': 0.321568627 ,
'ts\\_h': 0.325490196 ,
'ts`': 0.329411765 ,
'ts`_h': 0.333333333 ,
'u': 0.337254902 ,
'u:': 0.341176471 ,
'U@': 0.345098039 ,
'u@_n': 0.349019608 ,
'u@N': 0.352941176 ,
'ua': 0.356862745 ,
'ua_n': 0.360784314 ,
'uaI': 0.364705882 ,
'uAN': 0.368627451 ,
'uei': 0.37254902 ,
'UN': 0.376470588 ,
'uo': 0.380392157 ,
'V': 0.384313725 ,
'w': 0.388235294 ,
'x': 0.392156863 ,
'y': 0.396078431 ,
'y_n': 0.4 ,
'y{_n': 0.403921569 ,
'yE_r': 0.407843137 ,
'z': 0.411764706 ,
'z`': 0.415686275
}
dicty={
'-': 0   ,
'a': 1 ,
'o': 2 ,
'e': 3 ,
'i': 4 ,
'u': 5 ,
'v': 6 ,
'ai': 7 ,
'ei': 8 ,
'ao': 9 ,
'ou': 10 ,
'er': 11 ,
'an': 12 ,
'en': 13 ,
'ang': 14 ,
'ba': 15 ,
'bo': 16 ,
'bi': 17 ,
'bu': 18 ,
'bai': 19 ,
'bei': 20 ,
'bao': 21 ,
'bie': 22 ,
'ban': 23 ,
'ben': 24 ,
'bin': 25 ,
'bang': 26 ,
'beng': 27 ,
'bing': 28 ,
'pa': 29 ,
'po': 30 ,
'pi': 31 ,
'pu': 32 ,
'pai': 33 ,
'pei': 34 ,
'pao': 35 ,
'pou': 36 ,
'pie': 37 ,
'pan': 38 ,
'pen': 39 ,
'pin': 40 ,
'pang': 41 ,
'peng': 42 ,
'ping': 43 ,
'ma': 44 ,
'mo': 45 ,
'me': 46 ,
'mi': 47 ,
'mu': 48 ,
'mai': 49 ,
'mei': 50 ,
'mao': 51 ,
'mou': 52 ,
'miu': 53 ,
'mie': 54 ,
'man': 55 ,
'men': 56 ,
'min': 57 ,
'mang': 58 ,
'meng': 59 ,
'ming': 60 ,
'fa': 61 ,
'fo': 62 ,
'fu': 63 ,
'fei': 64 ,
'fou': 65 ,
'fan': 66 ,
'fen': 67 ,
'fang': 68 ,
'feng': 69 ,
'da': 70 ,
'de': 71 ,
'di': 72 ,
'du': 73 ,
'dai': 74 ,
'dei': 75 ,
'dui': 76 ,
'dao': 77 ,
'dou': 78 ,
'diu': 79 ,
'die': 80 ,
'dan': 81 ,
'dun': 82 ,
'dang': 83 ,
'deng': 84 ,
'ding': 85 ,
'dong': 86 ,
'ta': 87 ,
'te': 88 ,
'ti': 89 ,
'tu': 90 ,
'tai': 91 ,
'tui': 92 ,
'tao': 93 ,
'tou': 94 ,
'tie': 95 ,
'tan': 96 ,
'tun': 97 ,
'tang': 98 ,
'teng': 99 ,
'ting': 100 ,
'tong': 101 ,
'na': 102 ,
'ne': 103 ,
'ni': 104 ,
'nu': 105 ,
'nv': 106 ,
'nai': 107 ,
'nei': 108 ,
'nao': 109 ,
'nou': 110 ,
'niu': 111 ,
'nie': 112 ,
'nve': 113 ,
'nan': 114 ,
'nen': 115 ,
'nin': 116 ,
'nang': 117 ,
'neng': 118 ,
'ning': 119 ,
'nong': 120 ,
'la': 121 ,
'le': 122 ,
'li': 123 ,
'lu': 124 ,
'lv': 125 ,
'lai': 126 ,
'lei': 127 ,
'lao': 128 ,
'lou': 129 ,
'liu': 130 ,
'lie': 131 ,
'lve': 132 ,
'lan': 133 ,
'lin': 134 ,
'lun': 135 ,
'lang': 136 ,
'leng': 137 ,
'ling': 138 ,
'long': 139 ,
'ga': 140 ,
'ge': 141 ,
'gu': 142 ,
'gai': 143 ,
'gei': 144 ,
'gui': 145 ,
'gao': 146 ,
'gou': 147 ,
'gan': 148 ,
'gen': 149 ,
'gun': 150 ,
'gang': 151 ,
'geng': 152 ,
'gong': 153 ,
'ka': 154 ,
'ke': 155 ,
'ku': 156 ,
'kai': 157 ,
'kui': 158 ,
'kao': 159 ,
'kou': 160 ,
'kan': 161 ,
'ken': 162 ,
'kun': 163 ,
'kang': 164 ,
'keng': 165 ,
'kong': 166 ,
'ha': 167 ,
'he': 168 ,
'hu': 169 ,
'hai': 170 ,
'hei': 171 ,
'hui': 172 ,
'hao': 173 ,
'hou': 174 ,
'han': 175 ,
'hen': 176 ,
'hun': 177 ,
'hang': 178 ,
'heng': 179 ,
'hong': 180 ,
'ji': 181 ,
'ju': 182 ,
'jiu': 183 ,
'jie': 184 ,
'jue': 185 ,
'jin': 186 ,
'jun': 187 ,
'jing': 188 ,
'qi': 189 ,
'qu': 190 ,
'qiu': 191 ,
'qie': 192 ,
'que': 193 ,
'qin': 194 ,
'qun': 195 ,
'qing': 196 ,
'xi': 197 ,
'xu': 198 ,
'xiu': 199 ,
'xie': 200 ,
'xue': 201 ,
'xin': 202 ,
'xun': 203 ,
'xing': 204 ,
'za': 205 ,
'ze': 206 ,
'zi': 207 ,
'zu': 208 ,
'zai': 209 ,
'zei': 210 ,
'zui': 211 ,
'zao': 212 ,
'zou': 213 ,
'zan': 214 ,
'zen': 215 ,
'zun': 216 ,
'zang': 217 ,
'zeng': 218 ,
'zong': 219 ,
'ca': 220 ,
'ce': 221 ,
'ci': 222 ,
'cu': 223 ,
'cai': 224 ,
'cui': 225 ,
'cao': 226 ,
'cou': 227 ,
'can': 228 ,
'cen': 229 ,
'cun': 230 ,
'cang': 231 ,
'ceng': 232 ,
'cong': 233 ,
'sa': 234 ,
'se': 235 ,
'si': 236 ,
'su': 237 ,
'sai': 238 ,
'sui': 239 ,
'sao': 240 ,
'sou': 241 ,
'san': 242 ,
'sen': 243 ,
'sun': 244 ,
'sang': 245 ,
'seng': 246 ,
'song': 247 ,
'zha': 248 ,
'zhe': 249 ,
'zhi': 250 ,
'zhu': 251 ,
'zhai': 252 ,
'zhei': 253 ,
'zhui': 254 ,
'zhao': 255 ,
'zhou': 256 ,
'zhan': 257 ,
'zhen': 258 ,
'zhun': 259 ,
'zhang': 260 ,
'zheng': 261 ,
'zhong': 262 ,
'cha': 263 ,
'che': 264 ,
'chi': 265 ,
'chu': 266 ,
'chai': 267 ,
'chui': 268 ,
'chao': 269 ,
'chou': 270 ,
'chan': 271 ,
'chen': 272 ,
'chun': 273 ,
'chang': 274 ,
'cheng': 275 ,
'chong': 276 ,
'sha': 277 ,
'she': 278 ,
'shi': 279 ,
'shu': 280 ,
'shai': 281 ,
'shei': 282 ,
'shui': 283 ,
'shao': 284 ,
'shou': 285 ,
'shan': 286 ,
'shen': 287 ,
'shun': 288 ,
'shang': 289 ,
'sheng': 290 ,
're': 291 ,
'ri': 292 ,
'ru': 293 ,
'rui': 294 ,
'rao': 295 ,
'rou': 296 ,
'ran': 297 ,
'ren': 298 ,
'run': 299 ,
'rang': 300 ,
'reng': 301 ,
'rong': 302 ,
'ya': 303 ,
'ye': 304 ,
'yi': 305 ,
'yu': 306 ,
'yao': 307 ,
'you': 308 ,
'yue': 309 ,
'yan': 310 ,
'yin': 311 ,
'yun': 312 ,
'yang': 313 ,
'ying': 314 ,
'yong': 315 ,
'wa': 316 ,
'wo': 317 ,
'wu': 318 ,
'wai': 319 ,
'wei': 320 ,
'wan': 321 ,
'wen': 322 ,
'wang': 323 ,
'weng': 324 ,
'biao': 325 ,
'bian': 326 ,
'piao': 327 ,
'pian': 328 ,
'miao': 329 ,
'mian': 330 ,
'diao': 331 ,
'dian': 332 ,
'duo': 333 ,
'duan': 334 ,
'tiao': 335 ,
'tian': 336 ,
'tuo': 337 ,
'tuan': 338 ,
'niao': 339 ,
'nian': 340 ,
'niang': 341 ,
'nuo': 342 ,
'nuan': 343 ,
'lia': 344 ,
'liao': 345 ,
'lian': 346 ,
'liang': 347 ,
'luo': 348 ,
'luan': 349 ,
'gua': 350 ,
'guo': 351 ,
'guai': 352 ,
'guan': 353 ,
'guang': 354 ,
'kua': 355 ,
'kuo': 356 ,
'kuai': 357 ,
'kuan': 358 ,
'kuang': 359 ,
'hua': 360 ,
'huo': 361 ,
'huai': 362 ,
'huan': 363 ,
'huang': 364 ,
'jia': 365 ,
'jiao': 366 ,
'jian': 367 ,
'jiang': 368 ,
'jiong': 369 ,
'juan': 370 ,
'qia': 371 ,
'qiao': 372 ,
'qian': 373 ,
'qiang': 374 ,
'qiong': 375 ,
'quan': 376 ,
'xia': 377 ,
'xiao': 378 ,
'xian': 379 ,
'xiang': 380 ,
'xiong': 381 ,
'xuan': 382 ,
'zuo': 383 ,
'zuan': 384 ,
'cuo': 385 ,
'cuan': 386 ,
'suo': 387 ,
'suan': 388 ,
'zhua': 389 ,
'zhuo': 390 ,
'zhuai': 391 ,
'zhuan': 392 ,
'zhuang': 393 ,
'chua': 394 ,
'chuo': 395 ,
'chuai': 396 ,
'chuan': 397 ,
'chuang': 398 ,
'shua': 399 ,
'shuo': 400 ,
'shuai': 401 ,
'shuan': 402 ,
'shuang': 403 ,
'ruo': 404 ,
'ruan': 405 ,
'yuan': 406
}
def vsqx2notes(path):
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    trackElements=root.getElementsByTagName('vsTrack')
    masterElement=root.getElementsByTagName('masterTrack')[0]
    resolutionElement=masterElement.getElementsByTagName('resolution')[0]
    resolution=int(resolutionElement.childNodes[0].data)
    preElement=masterElement.getElementsByTagName('preMeasure')[0]
    pre=int(preElement.childNodes[0].data)
    timeSigElement=masterElement.getElementsByTagName('timeSig')[0]
    timeSig=(int(timeSigElement.getElementsByTagName('nu')[0].childNodes[0].data),
                int(timeSigElement.getElementsByTagName('de')[0].childNodes[0].data))
    tempoElement=masterElement.getElementsByTagName('tempo')[0]
    tempo=int(tempoElement.getElementsByTagName('v')[0].childNodes[0].data)
    mspt=(60000.0/tempo*100)/resolution
    jj=0
    for trackElement in trackElements:
        track=[]
        partElements=trackElement.getElementsByTagName('vsPart')
        for partElement in partElements:
            pt=int(partElement.getElementsByTagName('t')[0].childNodes[0].data)
            noteElements=partElement.getElementsByTagName('note')
            for noteElement in noteElements:
                tElement=noteElement.getElementsByTagName('t')[0]
                t=int(tElement.childNodes[0].data)
                durElement=noteElement.getElementsByTagName('dur')[0]
                dur=int(durElement.childNodes[0].data)
                nElement=noteElement.getElementsByTagName('n')[0]
                n=int(nElement.childNodes[0].data)
                lrcElement=noteElement.getElementsByTagName('y')[0]
                lrc=lrcElement.childNodes[0].data.lower()
                ipaElement=noteElement.getElementsByTagName('p')[0]
                ipa=ipaElement.childNodes[0].data
                track.append((pt+t,dur,n,lrc,ipa))
        # last=0
        # lastn=0
        # track.append((0,0,0,'',''))
        # for i in range(len(track)-1):
        #     #print(note)
        #     note=track[i]
        #     for j in range(int((note[0]-last)*mspt/10)):
        #         fout.write('0\n')
        #     rest=int((note[0]-last)*mspt)-int((note[0]-last)*mspt/10)*10
        #     for j in range(int((note[1]) * mspt / 10 / 4 * 5)):
        #         fout.write('%d %d %d %f '%(track[i+1][2],note[2],lastn,j/((note[1])*mspt/10)))
        #         try:
        #             fout.write('%d '%dicty[note[3].strip()])
        #         except KeyError:
        #             print('KeyError %s'%note[3].strip())
        #         fout.write('\n')
        #     rest=int((note[1])*mspt)-int((note[1])*mspt/10)*10
        #     last=note[0]+note[1]
        #     lastn=note[2]
        # jj+=1
        # fout.close()
        notes=[]
        end=0
        begin=23333333333333333333333
        for note in track:
            notes.append((int(note[0]*mspt/8),
            int((note[0]+note[1])*mspt/8),
            note[3],
            note[2]-49))
            begin=min(begin,int(note[0]*mspt/8))
            end = max(end, int((note[0] + note[1]) * mspt / 8))
        return notes,begin,end