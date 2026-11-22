"""CPU ray renderer. Coordinates and depth are in metres; no external 3D engine."""
import math
from pathlib import Path
import cv2
import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parent.parent

@njit(inline='always')
def box_hit(ox, oy, oz, dx, dy, dz, box):
    near, far = 0.001, 1e9
    axis, sign = 0, 1.0
    for k in range(3):
        origin = (ox, oy, oz)[k]; direction = (dx, dy, dz)[k]
        if abs(direction) < 1e-9:
            if origin < box[k] or origin > box[k+3]: return 1e9, 0, 0.0
        else:
            a = (box[k]-origin)/direction; b = (box[k+3]-origin)/direction
            s = -1.0
            if a > b: a, b, s = b, a, 1.0
            if a > near: near, axis, sign = a, k, s
            far = min(far, b)
            if far < near: return 1e9, 0, 0.0
    return near, axis, sign

@njit(parallel=True, cache=True)
def render_cpu(width, height, pos, forward, right, up, intrinsics, surfaces, textures, sizes, boxes, colors, triangles, tri_colors, ranges, surface_ids, box_ids, robot, robot_visible):
    image = np.empty((height, width, 3), np.uint8)
    depth = np.empty((height, width), np.float32)
    index = np.zeros((height, width), np.int32)
    for py in prange(height):
        for px in range(width):
            sx = (px+0.5-intrinsics[2])/intrinsics[0]
            sy = (intrinsics[3]-py-0.5)/intrinsics[1]
            dx = forward[0]+sx*right[0]+sy*up[0]; dy = forward[1]+sx*right[1]+sy*up[1]; dz = forward[2]+sx*right[2]+sy*up[2]
            inv = 1/math.sqrt(dx*dx+dy*dy+dz*dz); dx*=inv; dy*=inv; dz*=inv
            best=1e9; cr=95.; cg=120.; cb=145.; nx=0.; ny=0.; nz=1.; kind=-1; ident=0
            if dz < -1e-9:
                for i in range(surfaces.shape[0]):
                    s=surfaces[i]; t=(s[4]-pos[2])/dz
                    x=pos[0]+t*dx; y=pos[1]+t*dy
                    c=math.cos(-s[5]); sn=math.sin(-s[5]); cx=(s[0]+s[2])/2; cy=(s[1]+s[3])/2
                    lx=c*(x-cx)-sn*(y-cy); ly=sn*(x-cx)+c*(y-cy)
                    if t>0.001 and t<best and abs(lx)<=(s[2]-s[0])/2 and abs(ly)<=(s[3]-s[1])/2:
                        best=t; kind=i; ident=surface_ids[i]; nx=0.; ny=0.; nz=1.
                        u=int((x/sizes[i] % 1)*textures.shape[2]) % textures.shape[2]; v=int((y/sizes[i] % 1)*textures.shape[1]) % textures.shape[1]
                        cr=float(textures[i,v,u,0]); cg=float(textures[i,v,u,1]); cb=float(textures[i,v,u,2])
            for i in range(boxes.shape[0]):
                t, ax, sg=box_hit(pos[0],pos[1],pos[2],dx,dy,dz,boxes[i])
                if ranges[i,1]>ranges[i,0] and t<best:
                    for j in range(ranges[i,0],ranges[i,1]):
                        v=triangles[j]; e1=v[1]-v[0]; e2=v[2]-v[0]
                        p0=dy*e2[2]-dz*e2[1]; p1=dz*e2[0]-dx*e2[2]; p2=dx*e2[1]-dy*e2[0]
                        det=e1[0]*p0+e1[1]*p1+e1[2]*p2
                        if abs(det)<1e-10: continue
                        invdet=1./det; tx=pos[0]-v[0,0]; ty=pos[1]-v[0,1]; tz=pos[2]-v[0,2]
                        uu=(tx*p0+ty*p1+tz*p2)*invdet
                        if uu<0. or uu>1.: continue
                        q0=ty*e1[2]-tz*e1[1]; q1=tz*e1[0]-tx*e1[2]; q2=tx*e1[1]-ty*e1[0]
                        vv=(dx*q0+dy*q1+dz*q2)*invdet
                        if vv<0. or uu+vv>1.: continue
                        tt=(e2[0]*q0+e2[1]*q1+e2[2]*q2)*invdet
                        if tt>0.001 and tt<best:
                            best=tt; kind=-4; ident=box_ids[i]
                            nx=e1[1]*e2[2]-e1[2]*e2[1]; ny=e1[2]*e2[0]-e1[0]*e2[2]; nz=e1[0]*e2[1]-e1[1]*e2[0]
                            norm=math.sqrt(nx*nx+ny*ny+nz*nz)+1e-12; nx/=norm; ny/=norm; nz/=norm
                            if nx*dx+ny*dy+nz*dz>0: nx=-nx; ny=-ny; nz=-nz
                            cr=tri_colors[j,0]; cg=tri_colors[j,1]; cb=tri_colors[j,2]
                    continue
                if t<best:
                    best=t; kind=-2; ident=box_ids[i]; cr=colors[i,0]; cg=colors[i,1]; cb=colors[i,2]
                    nx=sg if ax==0 else 0.; ny=sg if ax==1 else 0.; nz=sg if ax==2 else 0.
            if robot_visible:
                # Vertical cylinder for the vacuum body.
                ox=pos[0]-robot[0]; oy=pos[1]-robot[1]; a=dx*dx+dy*dy; b=2*(ox*dx+oy*dy); c=ox*ox+oy*oy-0.175**2
                disc=b*b-4*a*c
                if a>1e-12 and disc>=0:
                    t=(-b-math.sqrt(disc))/(2*a); z=pos[2]+t*dz
                    if t>0.001 and t<best and 0.015<=z<=0.105:
                        best=t; kind=-3; nx=(ox+t*dx)/0.175; ny=(oy+t*dy)/0.175; nz=0.; cr=45.;cg=57.;cb=69.
                if abs(dz)>1e-9:
                    t=(0.105-pos[2])/dz; x=ox+t*dx; y=oy+t*dy
                    if t>0.001 and t<best and x*x+y*y<=0.175**2:
                        best=t;kind=-3;nx=0.;ny=0.;nz=1.;cr=72.;cg=90.;cb=104.
                        fx=-math.sin(robot[2]); fy=math.cos(robot[2]); along=x*fx+y*fy; side=x*fy-y*fx
                        if 0.07<along<0.15 and abs(side)<0.035: cr=36.;cg=190.;cb=210.
            if best<1e8:
                light=max(0., nx*(-0.35)+ny*(-0.2)+nz*0.915)
                shade=0.38+0.62*light
                # Hard shadow from scene boxes toward the directional light.
                hx=pos[0]+best*dx+nx*0.002; hy=pos[1]+best*dy+ny*0.002; hz=pos[2]+best*dz+nz*0.002
                for i in range(boxes.shape[0]):
                    t, _, _=box_hit(hx,hy,hz,-0.35,-0.2,0.915,boxes[i])
                    if t<1e8: shade*=0.60; break
                image[py,px,0]=min(255,int(cr*shade)); image[py,px,1]=min(255,int(cg*shade)); image[py,px,2]=min(255,int(cb*shade))
                depth[py,px]=best
            else:
                image[py,px,0]=int(cr); image[py,px,1]=int(cg); image[py,px,2]=int(cb); depth[py,px]=np.inf
            index[py,px]=ident
    return image, depth, index

def prepare(scene):
    import json
    cfg=json.loads((ROOT/'sim/assets/surfaces.json').read_text())['surfaces']
    surfs=[]; textures=[]; sizes=[]
    for i,s in enumerate(scene['surfaces']):
        surfs.append([*s['rect'], s.get('z',0)+i*0.0001+s.get('thickness',0), s.get('rot',0)])
        im=cv2.imread(str(ROOT/'sim/assets'/cfg[s['type']]['texture']))
        if im is None: raise FileNotFoundError(s['type'])
        textures.append(cv2.cvtColor(cv2.resize(im,(256,256)),cv2.COLOR_BGR2RGB)); sizes.append(cfg[s['type']]['size_m'])
    boxes=[]; colors=[]; triangles=[]; tri_colors=[]; ranges=[]; count=0
    for p in scene.get('props',[]):
        # The source supports yaw; bounding boxes are approximate when rotated.
        w,d,h=p['w'],p['d'],p['h']; yaw=p.get('yaw',0); w,d=abs(w*math.cos(yaw))+abs(d*math.sin(yaw)),abs(w*math.sin(yaw))+abs(d*math.cos(yaw))
        boxes.append([p['x']-w/2,p['y']-d/2,0,p['x']+w/2,p['y']+d/2,h]); colors.append(np.array(p.get('color',[0.3]*3))*255); ranges.append([count,count])
    for o in scene.get('objects',[]):
        if 'dims' in o: dims=o['dims']; lo=np.array([-dims[0]/2,-dims[1]/2,0.]); hi=lo+np.asarray(dims); ranges.append([count,count])
        else:
            mesh=Path(o['mesh']) if 'mesh' in o else ROOT/'sim/assets/objects'/f"{o['id']}.npz"
            with np.load(mesh) as data:
                verts=data['verts'].astype(float)*o.get('scale',1); faces=data['faces']; cols=data['colors']; yaw=o.get('yaw',0.); c,s=math.cos(yaw),math.sin(yaw)
                # Voxel clustering retains measured object silhouettes while keeping
                # the CPU triangle budget bounded; collision uses its bounding box.
                grid=np.floor((verts-verts.min(axis=0))/0.008).astype(np.int32)
                _, inverse=np.unique(grid,axis=0,return_inverse=True)
                counts=np.bincount(inverse); n=len(counts)
                reduced=np.column_stack([np.bincount(inverse,weights=verts[:,k],minlength=n)/counts for k in range(3)])
                reduced_colors=np.column_stack([np.bincount(inverse,weights=cols[:,k],minlength=n)/counts for k in range(3)])
                faces=inverse[faces]; faces=faces[(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,0]!=faces[:,2])]
                verts,cols=reduced,reduced_colors
                verts[:,:2]=verts[:,:2]@np.array([[c,s],[-s,c]])
                lo=verts.min(axis=0); hi=verts.max(axis=0); verts+=np.array([o['x'],o['y'],o.get('z',0)])
                triangles.extend(verts[faces]); tri_colors.extend(cols[faces].mean(axis=1)*255); ranges.append([count,count+len(faces)]); count+=len(faces)
        offset=np.array([o['x'],o['y'],o.get('z',0)]); boxes.append([*(lo+offset),*(hi+offset)]); colors.append(np.array(o.get('color',[0.75,0.42,0.15]))*255)
    surface_ids=np.array([100+{'tile':0,'laminate':1,'carpet':2,'shag':3}[s['type']] for s in scene['surfaces']],np.int32)
    box_ids=np.array([99]*len(scene.get('props',[]))+list(range(1,len(scene.get('objects',[]))+1)),np.int32)
    return np.asarray(surfs,np.float64),np.asarray(textures,np.uint8),np.asarray(sizes,np.float64),np.asarray(boxes,np.float64).reshape(-1,6),np.asarray(colors,np.float64).reshape(-1,3),np.asarray(triangles,np.float64).reshape(-1,3,3),np.asarray(tri_colors,np.float64).reshape(-1,3),np.asarray(ranges,np.int64).reshape(-1,2),surface_ids,box_ids

def camera(state, mode, free=None):
    x,y,yaw=state['x'],state['y'],state.get('yaw',0.0); fx,fy=-math.sin(yaw),math.cos(yaw)
    if mode=='third': pos=np.array([x-fx*0.85,y-fy*0.85,0.65]); target=np.array([x+fx*0.35,y+fy*0.35,0.05])
    elif mode=='top': pos=np.array([0.,0.5,8.]); target=np.array([0.,0.5,0.])
    elif mode=='robot': pos=np.array([x+fx*0.16,y+fy*0.16,0.13]); target=pos+np.array([fx,fy,-0.22])
    else: pos,target=free
    f=target-pos; f=f/np.linalg.norm(f); worldup=np.array([0.,1.,0.]) if abs(f[2])>0.999 else np.array([0.,0.,1.]); r=np.cross(f,worldup); r/=np.linalg.norm(r); u=np.cross(r,f)
    return pos,f,r,u

def render(scene_data, state, mode, width, height, free=None, calibration=None, with_index=False):
    pos,f,r,u=camera(state,mode,free)
    focal=height/(2*math.tan(math.pi/6))
    intrinsic=np.array([focal,focal,width/2,height/2])
    if mode=='robot' and calibration is not None:
        plane=np.array(calibration['floor_plane'],float); n=plane[:3]/np.linalg.norm(plane[:3]); d=plane[3]/np.linalg.norm(plane[:3])
        if d<0: n=-n; d=-d
        fwd=np.array([0.,0.,1.])-n*n[2]; fwd/=np.linalg.norm(fwd); right=np.cross(fwd,n)
        R=np.stack([right,fwd,n]); yaw=state.get('yaw',0.); c,s=math.cos(yaw),math.sin(yaw); Rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        R=Rz@R; pos=np.array([state['x'],state['y'],d]); f=R[:,2]; r=R[:,0]; u=-R[:,1]
        intrinsic=np.array([calibration['fx']*width/calibration['width'],calibration['fy']*height/calibration['height'],calibration['cx']*width/calibration['width'],calibration['cy']*height/calibration['height']])
    rgb, depth, index=render_cpu(width,height,pos,f,r,u,intrinsic,*scene_data,np.array([state['x'],state['y'],state.get('yaw',0.)]),mode!='robot')
    return (rgb, depth, index) if with_index else (rgb, depth)
