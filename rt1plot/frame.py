import matplotlib.patches as patches
import numpy as np
import dxfgrabber
import cv2

class Frame():
    def __init__(self,dxf_file:str):
        dxf = dxfgrabber.readfile(dxf_file)
        print("DXF version: {}".format(dxf.dxfversion))
        self.header_var_count = len(dxf.header) # dict of dxf header vars
        self.layer_count = len(dxf.layers) # collection of layer definitions
        self.block_definition_count = len(dxf.blocks) #  dict like collection of block definitions
        self.entity_count = len(dxf.entities) # list like collection of entities
        self.all_lines = [entity for entity in dxf.entities if entity.dxftype == 'LINE']
        self.all_circles = [entity for entity in dxf.entities if entity.dxftype == 'CIRCLE']
        self.all_arcs = [entity for entity in dxf.entities if entity.dxftype == 'ARC']
        print('num of lines: ',len(self.all_lines))
        print('num of circs: ',len(self.all_circles))
        print('num of arcs : ',len(self.all_arcs))

    def append_frame(self,ax,label:bool=False,**kwargs):

        default_kwargs = {"linewidth":1, "alpha":1.0, "color":'gray'}
        default_kwargs.update(kwargs)
        kwargs = default_kwargs
        
        #線をプロット
        for i in range(len(self.all_lines)):
            ax.plot([self.all_lines[i].start[0]/1000, self.all_lines[i].end[0]/1000],
                    [self.all_lines[i].start[1]/1000, self.all_lines[i].end[1]/1000],
                    **kwargs)
            if label:
                ax.text(self.all_lines[i].start[0]/1000, self.all_lines[i].start[1]/1000, "l."+str(i), size = 10, color = "blue")

        #f弧をプロット
        for i in range(len(self.all_arcs)):   
            ax.add_patch(
                patches.Arc(
                    xy     =(self.all_arcs[i].center[0]/1000, self.all_arcs[i].center[1]/1000),
                    width  = self.all_arcs[i].radius*2/1000,
                    height = self.all_arcs[i].radius*2/1000,
                    theta1 = self.all_arcs[i].start_angle,
                    theta2 = self.all_arcs[i].end_angle,
                    **kwargs)

             ) 
            if label:
                x = self.all_arcs[i].center[0]/1000+  self.all_arcs[i].radius/1000*np.cos(np.pi* self.all_arcs[i].start_angle/180)
                y = self.all_arcs[i].center[1]/1000+  self.all_arcs[i].radius/1000*np.sin(np.pi* self.all_arcs[i].start_angle/180)
                ax.text(x, y, "a."+str(i), size = 10, color = "red")

    def grid_input(self,R,Z,fill_point=(0.5,0)):
        h,w = Z.size,R.size
        R_extend = np.empty(R.size+1)
        Z_extend = np.empty(Z.size+1)
        R_extend[0] =  R[0]  - 0.5* (R[1]-R[0])
        R_extend[-1] = R[-1] + 0.5* (R[-1]-R[-2])
        R_extend[1:-1] = 0.5 * (R[:-1] + R[1:])
        Z_extend[0] =  Z[0]  - 0.5* (Z[1]-Z[0])
        Z_extend[-1] = Z[-1] + 0.5* (Z[-1]-Z[-2])
        Z_extend[1:-1] = 0.5 * (Z[:-1] + Z[1:])


        RR,ZZ = np.meshgrid(R_extend,Z_extend,indexing='xy')

        R_tr = RR[+1:,+1:]
        Z_tr = ZZ[+1:,+1:]
        R_tl = RR[+1:,:-1]
        Z_tl = ZZ[+1:,:-1]
        R_br = RR[:-1,+1:]
        Z_br = ZZ[:-1,+1:]
        R_bl = RR[:-1,:-1]
        Z_bl = ZZ[:-1,:-1]

        list1 = ['top','bottom','left','right']

        self.Is_bound = np.zeros((Z.size,R.size),np.bool)
        
        for i in range(len(self.all_lines)):
            R4, Z4 = self.all_lines[i].start[0]/1000, self.all_lines[i].start[1]/1000
            R3, Z3 = self.all_lines[i].end[0]/1000, self.all_lines[i].end[1]/1000
            for mode in list1:
                if mode == 'top':
                    R1 = R_tr
                    R2 = R_tl
                    Z1 = Z_tr 
                    Z2 = Z_tl
                if mode == 'bottom':
                    R1 = R_br
                    R2 = R_bl
                    Z1 = Z_br 
                    Z2 = Z_bl
                if mode == 'right':
                    R1 = R_tr
                    R2 = R_br
                    Z1 = Z_tr 
                    Z2 = Z_br
                if mode == 'left':
                    R1 = R_tl
                    R2 = R_bl
                    Z1 = Z_tl
                    Z2 = Z_bl
            
                D = (R4-R3) * (Z2-Z1) - (R2-R1) * (Z4-Z3)
                W1, W2 = Z3*R4-Z4*R3, Z1*R2 - Z2*R1

                D_is_0 = (D <= 1e-100)

                D += D_is_0 * 1e-100
            
                R_inter = ( (R2-R1) * W1 - (R4-R3) * W2 ) / D
                Z_inter = ( (Z2-Z1) * W1 - (Z4-Z3) * W2 ) / D
                del W1,W2,D
                
                is_in_Rray_range = (R2 - R_inter) * (R1 - R_inter) <= 1.e-10 
                is_in_Zray_range = (Z2 - Z_inter) * (Z1 - Z_inter) <= 1.e-10 
                is_in_Rfra_range = (R4 - R_inter) * (R3 - R_inter) <= 1.e-10 
                is_in_Zfra_range = (Z4 - Z_inter) * (Z3 - Z_inter) <= 1.e-10 
                is_in_range =  np.logical_and(is_in_Rray_range,is_in_Zray_range) * np.logical_and(is_in_Rfra_range,is_in_Zfra_range) 
                # 水平や垂直  な線に対応するため

                self.Is_bound  += is_in_range

        

        for i in range(len(self.all_arcs)):
            Rc, Zc =(self.all_arcs[i].center[0]/1000, self.all_arcs[i].center[1]/1000)
            radius = self.all_arcs[i].radius/1000
            sta_angle, end_angle = self.all_arcs[i].start_angle ,self.all_arcs[i].end_angle 

            
            for mode in list1:
                if mode == 'top':   
                    R1 = R_tr
                    R2 = R_tl
                    Z1 = Z_tr 
                    Z2 = Z_tl
                if mode == 'bottom':
                    R1 = R_br
                    R2 = R_bl
                    Z1 = Z_br 
                    Z2 = Z_bl
                if mode == 'right':
                    R1 = R_tr
                    R2 = R_br
                    Z1 = Z_tr 
                    Z2 = Z_br
                if mode == 'left':
                    R1 = R_tl
                    R2 = R_bl
                    Z1 = Z_tl
                    Z2 = Z_bl
                    
                lR = R2-R1
                lZ = Z2-Z1
                S  = R2*Z1 - R1*Z2      

                D = (lR**2+lZ**2)*radius**2 + 2*lR*lZ*Rc*Zc - 2*(lZ*Rc-lR*Zc)*S - lR**2 *Zc**2 -lZ**2*Rc**2-S**2 #判別式
                exist = D > 0

                Ri1 = (lR**2 *Rc + lR*lZ *Zc - lZ *S + lR * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# １つ目の交点のR座標
                Zi1 = (lZ**2 *Zc + lR*lZ *Rc + lR *S + lZ * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# １つ目の交点のZ座標
                Ri2 = (lR**2 *Rc + lR*lZ *Zc - lZ *S - lR * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# 2つ目の交点のR座標
                Zi2 = (lZ**2 *Zc + lR*lZ *Rc + lR *S - lZ * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# 2つ目の交点のZ座標
                del D, exist

                is_in_ray_range1  = np.logical_and((R2 - Ri1) * (R1 - Ri1) <= 1e-10 ,(Z2 - Zi1) * (Z1- Zi1) <= 1e-10) # 交点1が線分内にあるか判定
                is_in_ray_range2  = np.logical_and((R2 - Ri2) * (R1 - Ri2) <= 1e-10 ,(Z2 - Zi2) * (Z1- Zi2) <= 1e-10) # 交点2が線分内にあるか判定


                cos1 = (Ri1-Rc) / radius
                sin1 = (Zi1-Zc) / radius
                atan = np.arctan2(sin1,cos1)
                theta1 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)
                cos2 = (Ri2-Rc) / radius    
                sin2 = (Zi2-Zc) / radius 
                atan = np.arctan2(sin2,cos2)
                theta2 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)

                del cos1,sin1,atan,cos2,sin2

                is_in_arc1 =  (end_angle - theta1) * (sta_angle - theta1) * (end_angle-sta_angle) <= 0 # 交点1が弧の範囲内あるか判定
                is_in_arc2 =  (end_angle - theta2) * (sta_angle - theta2) * (end_angle-sta_angle) <= 0 # 交点1が弧の範囲内あるか判定

                is_real_intercept1 = is_in_ray_range1 * is_in_arc1
                is_real_intercept2 = is_in_ray_range2 * is_in_arc2
                is_real_intercept  = is_real_intercept1 + is_real_intercept2
                
                self.Is_bound += is_real_intercept
        
        mask = np.zeros((h,w), np.uint8)


        # 塗りつぶしの開始インデクスを探索
        i_r,i_z = 0,0
        for i in range(R.size-1):
            if (R[i] - fill_point[0])*(R[i+1] - fill_point[0]  ) < 0:
                i_r =  i 
                break 
        for i in range(Z.size-1):
            if (Z[i] - fill_point[1])*(Z[i+1] - fill_point[1]  ) < 0:
                i_z =  i 
                break 

        print(i_r,i_z)

        self.fill =  np.zeros((h,w), np.uint8)
        self.fill[:,:] = 1 *self.Is_bound[:,:]                        
        mask = np.zeros((h+2, w+2), np.uint8)
        
        cv2.floodFill(self.fill, mask, (i_r,i_z), 2)

        self.Is_in = (self.fill == 2)
        self.Is_out = (self.fill == 0)
        self.NaN_factor = np.where(self.fill ==2 , 1.0, np.nan)
        self.imshow_extent = (R_extend[0],R_extend[-1],Z_extend[0],Z_extend[-1])