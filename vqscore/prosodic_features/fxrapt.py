import math
import torch
import torch.nn.functional as F
import numpy as np

from scipy.linalg import toeplitz
from torch.nn.functional import conv1d


def fxrapt(signal, fs, mode='g'):
    """
    FXRAPT RAPT pitch tracker [FX,VUV]=(SIGNAL,FS,M)
    
     Input:   signal(ns) Speech signal
              fs         Sample frequency (Hz)
              mode       'g' will plot a graph [default if no output arguments]
                         'u' will include unvoiced fames (with fx=NaN)
    
     Outputs: fx(nframe)     Larynx frequency for each fram,e (or NaN for silent/unvoiced)
              tt(nframe,3)  Start and end samples of each frame. tt(*,3)=1 at the start of each talk spurt
    
     Plots a graph if no outputs are specified showing lag candidates and selected path
    

     Bugs/Suggestions:
       (1) Include backward DP pass and output the true cost for each candidate.
       (2) Add an extra state to distinguish between voiceless and silent
       (3) N-best DP to allow longer term penalties (e.g. for frequent pitch doubling/halving)

     The algorithm is taken from [1] with the following differences:
    
          (a)  the factor AFACT which in the Talkin algorithm corresponds roughly
               to the absolute level of harmonic noise in the correlation window. This value
               is here calculated as the maximum of three figures:
                       (i) an absolute floor set by p.absnoise
                      (ii) a multiple of the peak signal set by p.signoise
                     (iii) a multiple of the noise floor set by p.relnoise
          (b) The LPC used in calculating the Itakura distance uses a Hamming window rather than
              a Hanning window.
    
     A C implementation of this algorithm by Derek Lin and David Talkin is included as  "get_f0.c"
     in the esps.zip package available from http://www.speech.kth.se/esps/esps.zip under the BSD
     license.
    
     Refs:
          [1]   D. Talkin, "A Robust Algorithm for Pitch Tracking (RAPT)"
                in "Speech Coding & Synthesis", W B Kleijn, K K Paliwal eds,
                Elsevier ISBN 0444821694, 1995

          Copyright (C) Mike Brookes 2006-2013
          Version: $Id: fxrapt.m 10185 2017-10-04 08:20:32Z dmb $
    
       VOICEBOX is a MATLAB toolbox for speech processing.
       Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   This program is free software; you can redistribute it and/or modify
    %   it under the terms of the GNU General Public License as published by
    %   the Free Software Foundation; either version 2 of the License, or
    %   (at your option) any later version.
    %
    %   This program is distributed in the hope that it will be useful,
    %   but WITHOUT ANY WARRANTY; without even the implied warranty of
    %   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    %   GNU General Public License for more details.
    %
    %   You can obtain a copy of the GNU General Public License from
    %   http://www.gnu.org/copyleft/gpl.html or by writing to
    %   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    signal = signal.flatten() # force s to be a column
    doback = 0   # don't do backwards DP for now

    # set default parameters
    p = {
        "f0min": 50,           # Min F0 (Hz)
        "f0max": 500,          # Max F0 (Hz)
        "tframe": 0.01,        # frame size (s)
        "tlpw": 0.005,         # low pass filter window size (s)
        "tcorw": 0.0075,       # correlation window size (s)
        "candtr": 0.3,         # minimum peak in NCCF
        "lagwt": 0.3,          # linear lag taper factor
        "freqwt": 0.02,        # cost factor for F0 change
        "vtranc": 0.005,       # fixed voice-state transition cost
        "vtrac": 0.5,          # delta amplitude modulated transition cost
        "vtrsc": 0.5,          # delta spectrum modulated transition cost
        "vobias": 0.0,         # bias to encourage voiced hypotheses
        "doublec": 0.35,       # cost of exact doubling or halving
        "absnoise": 0,         # absolute rms noise level
        "relnoise": 2,         # rms noise level relative to noise floor
        "signoise": 0.001,     # ratio of peak signal rms to noise floor (0.001: 60dB)
        "ncands": 20,          # max hypotheses at each frame
        "trms": 0.03,          # window length for rms measurement
        "dtrms": 0.02,         # window spacing for rms measurement
        "preemph": -7000,      # s-plane position of preemphasis zero
        "nfullag": 7,          # number of full lags to try (must be odd)
    }

    # derived parameters (mostly dependent on sample rate fs)

    krms = round(p['trms'] * fs)            # window length for rms measurement
    kdrms = round(p['dtrms'] * fs)          # window spacing for rms measurement
    rmswin = torch.hann_window(krms) ** 2
    kdsmp = round(0.25 * fs / p['f0max'])
    hlpw = round(p['tlpw'] * fs / 2)          # force window to be an odd length
    tmp = torch.round(torch.sinc(torch.arange(-hlpw, hlpw+1) / kdsmp), decimals=5)
    blp = tmp * torch.hamming_window(2 * hlpw + 1, periodic=False)
    fsd = fs / kdsmp
    kframed = round(fsd * p['tframe'])      # downsampled frame length
    kframe = kframed * kdsmp           # frame increment at full rate
    rmsix = torch.arange(krms) + math.floor((kdrms-kframe)/2) # rms index according to Talkin; better=(1:krms)+floor((kdrms-krms+1)/2)
    minlag = math.ceil(fsd / p['f0max'])
    maxlag = round(fsd / p['f0min'])        # use round() only because that is what Talkin does
    kcorwd=round(fsd * p['tcorw'])        # downsampled correlation window
    kcorw = kcorwd * kdsmp             # full rate correlation window
    spoff = max(hlpw - math.floor(kdsmp/2), 1 + kdrms - rmsix[0] - kframe - 1)  # offset for first speech frame at full rate
    sfoff = spoff - hlpw + math.floor(kdsmp/2) - 1    # offset for downsampling filter
    sfi = torch.arange(kcorwd)                   # initial decimated correlation window index array
    sfhi = torch.arange(kcorw)                   # initial correlation window index array
    sfj = torch.arange(kcorwd+maxlag)
    lagoff = (minlag - 1) * kdsmp        # lag offset when converting to high sample rate
    beta = p['lagwt'] * p['f0min'] / fs            # bias towards low lags
    log2 = math.log(2)
    lpcord = 2 + round(fs / 1000)        # lpc order for itakura distance
    hnfullag = math.floor(p['nfullag'] / 2)
    jumprat = math.exp((p['doublec'] + log2) / 2)  # lag ratio at which octave jump cost is lowest
    ssq = torch.pow(signal, 2)
    csssq = torch.cumsum(ssq, 0)
    #sqrt(min(csssq(kcorw+1:end)-csssq(1:end-kcorw))/kcorw);
    afact = max([p['absnoise']**2, torch.max(ssq) * p['signoise']**2, torch.min(csssq[kcorw:] - csssq[:-kcorw])*(p['relnoise']/kcorw)**2]) ** 2 * kcorw ** 2

    # downsample signal to approx 2 kHz to speed up autocorrelation calculation
    # kdsmp is the downsample factor

    sf = conv1d(signal[sfoff+1:][None], (blp/blp.sum())[None].fliplr()[None,:])[0]
    sp = conv1d(signal[None], torch.tensor([1, math.exp(p['preemph']/fs)])[None].fliplr()[None,:], padding=1)[0] # preemphasised speech for LPC calculation
    
    sf = sf[:-1:kdsmp]             # downsample to =~2kHz
    nsf = sf.nelement()            # length of downsampled speech
    ns = signal.nelement()         # length of full rate speech

    # Calculate the frame limit to ensure we don't run off the end of the speech or decimated speech:
    #   (a) For decimated autocorrelation when calculating sff():  (nframe-1)*kframed+kcorwd+maxlag <= nsf
    #   (b) For full rate autocorrelation when calculating sfh():  max(fho)+kcorw+maxlag*kdsamp+hnfllag <= ns
    #   (c) For rms ratio window when calculating rr            :  max(fho)+rmsix(end) <= ns
    # where max(fho) = (nframe-1)*kframe + spoff

    nframe = math.floor(1+min((nsf-kcorwd-maxlag)/kframed,(ns-spoff-max(kcorw-maxlag*kdsmp-hnfullag,rmsix[-1]))/kframe))

    # now search for autocorrelation peaks in the downsampled signal
    cost = torch.zeros(nframe, p['ncands']);      # cumulative cost
    prev = torch.zeros(nframe, p['ncands']);      # traceback pointer
    mcands = torch.zeros(nframe);            # number of actual candidates excluding voiceless
    lagval = torch.ones(nframe, p['ncands']-1) * torch.nan    # lag of each voiced candidate
    tv = torch.zeros(nframe, 6)             # diagnostics: 1=voiceless cost, 2=min voiced cost, 3:cumulative voiceless-min voiced
    if doback:
        costms = []

    # Main processing loop for each 10 ms frame
    for iframe in range(nframe):       # loop for each frame (~10 ms)
        # Find peaks in the normalized autocorrelation of subsampled (2Khz) speech
        # only keep peaks that are > 30% of highest peak
        fho = iframe * kframe + spoff
        sff = sf[iframe * kframed + sfj]
        sffdc = sff[sfi].mean()       # mean of initial correlation window length
        sff = sff - sffdc             # subtract off the mean
        nccfd = normxcor(sff[:kcorwd], sff[minlag:]);
        ipkd, vpkd = v_findpeaks(nccfd,'q')
        
        vipkd = torch.stack([vpkd, ipkd]).t()
        if vipkd.size(0) > 0:
            vipkd = vipkd[vpkd>=max(vpkd)*p['candtr'],:]          # eliminate peaks that are small
            if vipkd.size(0)>p['ncands']-1:
                vipkd = vipkd[vipkd[:,0].argsort(), :]
                vipkd = vipkd[vipkd.size(0)-p['ncands']:]   # eliminate lowest to leave only ncands-1
            lagcan = (vipkd[:,1] * kdsmp + lagoff).round().long()  # convert the lag candidate values to the full sample rate
            nlcan = lagcan.nelement()
        else:
            nlcan = 0
        
        # If there are any candidate lag values (nlcan>0) then refine their accuracy at the full sample rate
        if nlcan>0:
            sfh = signal[fho + torch.arange(0, kcorw+max(lagcan)+hnfullag)]
            sfhdc = sfh[sfhi].mean()
            sfh = sfh - sfhdc
            e0 = torch.pow(sfh[sfhi], 2).sum()     # energy of initial correlation window (only needed to store in tv(:,6)
            lagl2 = lagcan[None,:].expand(p['nfullag']+kcorw-1,-1) + torch.arange(1-hnfullag-1, hnfullag+kcorw)[:,None].expand(-1,nlcan)
            nccf = normxcor(sfh[:kcorw], sfh[lagl2], afact)
            
            maxcc, maxcci = nccf.max(0)
            nccf = nccf.t().flatten() # The t() is for replicating MATLAB's row first
            vipk = torch.stack([maxcc.flatten(), lagcan.flatten()+maxcci.flatten()-hnfullag]).t()
            vipk = vipk[:,[0, 1, 1]]
            maxccj = maxcci.flatten() + p['nfullag'] * torch.arange(0, nlcan)   # vector index into nccf array
            msk = ((maxcci+1) % (p['nfullag']-1) != 1) & \
                  ((2 * nccf[maxccj] - nccf[(maxccj-1) % (p['nfullag']*nlcan)] - nccf[(maxccj+1) % (p['nfullag']*nlcan)])>0)  # don't do quadratic interpolation for the end ones
            if torch.any(msk):
                maxccj = maxccj[msk]
                vipk[msk,2] += (nccf[maxccj+1]-nccf[maxccj-1])/(2*(2*nccf[maxccj]-nccf[maxccj-1]-nccf[maxccj+1]))            

            vipk = vipk[maxcc>=maxcc.max()*p['candtr'],:]          # eliminate peaks that are small
            if vipk.size(0)>p['ncands']-1:
                idx = vipk[:,0].argsort(descending=True)
                # vipk = vipk[idx][:p['ncands']+1]
                vipk = vipk[idx][:(p['ncands'] - 1)]  # Azam: Take at most ncands-1 candidates
            mc = vipk.size(0)
        else:
            mc = 0

        # We now have mc lag candidates at the full sample rate
        mc1 = mc + 1             # total number of candidates including "unvoiced" possibility
        mcands[iframe] = mc      # save number of lag candidates (needed for pitch consistency cost calculation)
        if mc>0:
            lagval[iframe,:mc] = vipk[:,2]
            cost[iframe,0] = p['vobias'] + max(vipk[:,0])   # voiceless cost
            cost[iframe,1:mc1] = 1 - vipk[:,0] * (1-beta*vipk[:,2])   # local voiced costs
            tv[iframe,1] = cost[iframe,1:mc1].min()
        else:
            cost[iframe,1] = p['vobias']     # if no lag candidates (mc=0), then the voiceless case is the only possibility

        tv[iframe,0] = cost[iframe,0]
        if iframe>0:                         # if it is not the first frame, then calculate pitch consistency and v/uv transition costs
            mcp = int(mcands[iframe-1])
            costm = torch.zeros(mcp+1, mc1)   # cost matrix: rows and cols correspond to candidates in previous and current frames (incl voiceless)
            
            # if both frames have at least one lag candidate, then calculate a pitch consistency cost
            if (mc*mcp)>0:
                lrat = torch.abs(torch.log(lagval[iframe,:mc][None,:].expand(mcp,-1)/lagval[iframe-1,:mcp][:,None].expand(-1,mc)))
                costm[1:,1:] = p['freqwt'] * torch.min(lrat, p['doublec'] + torch.abs(lrat-log2));  # allow pitch doubling/halving

            # if either frame has a lag candidate, then calculate the cost of voiced/voiceless transition and vice versa            
            if (mc+mcp)>0:
                max_idx = signal.size(0) - 1
                bounded_rmsix = torch.clamp(fho + rmsix, 0, max_idx)
                bounded_rmsix_kdrms = torch.clamp(fho + rmsix - kdrms, 0, max_idx)

                rr = torch.sqrt(torch.mm(rmswin[None, :], (signal[bounded_rmsix] ** 2)[:, None]) /
                                torch.mm(rmswin[None, :],(signal[bounded_rmsix_kdrms] ** 2)[:, None]))
                ss = 0.2 / (distitar(lpcauto(sp[bounded_rmsix], lpcord)[0], lpcauto(sp[bounded_rmsix_kdrms], lpcord)[0], 'e') - 0.8)

                # rr = torch.sqrt(torch.mm(rmswin[None,:],(signal[fho+rmsix]**2)[:,None]) /
                #                 torch.mm(rmswin[None,:],(signal[fho+rmsix-kdrms]**2)[:,None])) # amplitude "gradient"
                # ss = 0.2 / (distitar(lpcauto(sp[fho+rmsix], lpcord)[0], lpcauto(sp[fho+rmsix-kdrms], lpcord)[0], 'e') - 0.8)   # Spectral stationarity: note: Talkin uses Hanning instead of Hamming windows for LPC
                costm[0,1:] = p['vtranc'] + p['vtrsc'] * ss + p['vtranc'] / rr   # voiceless -> voiced cost
                costm[1:,0] = p['vtranc'] + p['vtrsc'] * ss + p['vtranc'] * rr
                tv[iframe,3:5] = torch.stack([costm[0,mc1-1], costm[mcp-1,0]])

            costm = costm + cost[iframe-1,:mcp+1][:,None]  # add in cumulative costs
            costi, previ = costm.min(0);
            cost[iframe,:mc1] = cost[iframe,:mc1] + costi
            prev[iframe,:mc1] = previ
        else:                            # first ever frame
            costm = torch.zeros(mc1)     # create a cost matrix in case doing a backward recursion
        
        if mc>0:
            tv[iframe,2] = cost[iframe,0] - cost[iframe,1:mc1].min()
            tv[iframe,5] = 5 * torch.log10(e0 * e0 / afact + 1e-10)
        
        if doback:
            costms.append([iframe, costms])      # need to add repmatted cost into this

    # now do traceback
    # best = torch.zeros(nframe)
    # best[nframe-1] = cost[nframe-1,:int(mcands[nframe-1])].argmin()
    # for i in range(nframe-1,1,-1):
    #     best[i-1] = prev[i,best[i].long()]

    # Azam: I added these 12 lines to check before attempting to find argmin
    # An occurs when the pitch detection algorithm fails to find any valid candidates for the last frame
    # now do traceback
    best = torch.zeros(nframe)
    # Check before attempting to find argmin
    if int(mcands[nframe - 1]) > 0 and cost[nframe - 1, :int(mcands[nframe - 1])].numel() > 0:
        best[nframe - 1] = cost[nframe - 1, :int(mcands[nframe - 1])].argmin()
    else:
        # Set to a default value or mark as unvoiced
        best[nframe - 1] = 0  # or another appropriate default for your algorithm

    # Continue with the traceback for other frames
    for i in range(nframe - 1, 0, -1):
        if best[i].item() < prev[i].size(0):  # Check if index is valid
            best[i - 1] = prev[i, best[i].long()]
        else:
            best[i - 1] = 0  # Default value

    vix = torch.where(best>0)[0]
    fx = torch.ones(nframe) * torch.nan     # unvoiced frames will be NaN
    
    if vix.nelement() > 0:
        fx[vix] = fs * lagval.t().flatten()[(vix+nframe*(best[vix]-1)).long()]**(-1) # leave as NaN if unvoiced

    tt = torch.zeros(nframe, 3)
    tt[:,0] = torch.arange(1, nframe+1) * kframe + spoff       # find frame times
    tt[:,1] = tt[:,0] + kframe - 1
    jratm = .5 * (jumprat+1/jumprat)
    tt[1:,2] = abs(fx[1:]/fx[:-1]-jratm)>jumprat - jratm    # new spurt if frequency ratio is outside (1/jumprat,jumprat)
    tt[1,2] = 1           # first frame always starts a spurt
    tt[torch.where(fx[:-1].isnan())[0],2] = 1 # NaN always forces a new spurt

    if not (mode=='u'):
        tt = tt[~fx.isnan()]    # remove NaN spurts
        fx = fx[~fx.isnan()]

    return fx, tt.long()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def lpcauto(s, p=12, t=None):
    s = s.flatten()  # Ensure column vector
    
    if t is None:
        t = torch.tensor([[len(s), len(s), 0]])
    else:
        t = torch.tensor(t, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
    
    nf, ng = t.shape
    if ng < 2:
        t = torch.cat((t, t[:, :1]), dim=1)
    if ng < 3:
        t = torch.cat((t, torch.zeros(nf, 1)), dim=1)
    
    if nf == 1:
        nf = int(1 + (len(s) - t[0, 1] - t[0, 2]) / t[0, 0])
        tr = 0
    else:
        tr = 1
    
    ar = torch.zeros((nf, p + 1))
    ar[:, 0] = 1
    e = torch.zeros(nf)
    
    t1 = 0
    it = 0
    nw = -1
    r = torch.arange(p + 1)
    k = torch.zeros((nf, 2), dtype=int)
    
    for jf in range(nf):
        k[jf, 0] = int(torch.ceil(t1 + t[it, 2]))
        k[jf, 1] = int(torch.ceil(t1 + t[it, 2] + t[it, 1] - 1))

        cs = torch.arange(k[jf, 0], k[jf, 1] + 1)
        nc = len(cs)
        pp = min(p, nc)
        dd = s[cs]
        
        if nc != nw:
            ww = torch.hamming_window(nc)
            nw = nc
            y = torch.zeros(nc + p)
            c = torch.arange(nc).unsqueeze(1)
        
        wd = dd * ww  # Windowed data vector
        y[:nc] = wd  # Data vector with p appended zeros
        z = torch.zeros((nc, pp + 1))
        
        # Creating the data matrix
        indices = c + r.unsqueeze(0)
        z[:] = y[indices]
        
        rr = wd @ z
        rm = torch.tensor(toeplitz(rr[:pp].numpy()))
        rk = torch.linalg.matrix_rank(rm)
        
        if rk:
            if rk < pp:
                rm = rm[:rk, :rk]
            ar[jf, 1:rk + 1] = -torch.linalg.solve(rm, rr[1:rk + 1])
        
        e[jf] = rr @ ar[jf, :pp + 1]
        t1 += t[it, 0]
        it += tr
    
    return ar, e, k


def normxcor(x,y,d=0):
    """
    # Calculate the normalized cross correlation of column vectors x and y
    # we can calculate this in two ways but fft is much faster even for nx small
    # We must have nx<=ny and the output length is ny-nx+1
    # note that this routine does not do mean subtraction even though this is normally a good idea
    # if y is a matrix, we correlate with each column
    # d is a constant added onto the normalization factor
    # v(j)=x'*yj/sqrt(d + x'*x * yj'*yj) where yj=y(j:j+nx-1) for j=1:ny-nx+1
    """
    def nextpow2(N):
        """ Function for finding the next power of 2 """
        n = 0
        val = 1
        while val < N:
            val *= 2
            n += 1
        return n

    x = x[:,None] if x.ndim==1 else x
    y = y[:,None] if y.ndim==1 else y
    nx = len(x)
    ny = y.size(0)
    my = y.size(1) if y.ndim == 2 else 1
    nv = 1 + ny - nx
    if nx>ny:
        print('second argument is shorter than the first');
        return

    nf = 2 ** nextpow2(ny)
    tmp = torch.conj(torch.fft.rfft(x,nf,0)).expand(-1,my) * torch.fft.rfft(y,nf,0)
    w = torch.fft.irfft(tmp, dim=0)
    s = torch.zeros(ny+1, my)
    s[1:,:] = torch.cumsum(y**2, 0)
    v = w[:nv,:]/(d + torch.mm(x.t(), x) * (s[nx:,:] - s[:-nx,:])).sqrt()
    return v


def distitar(ar1, ar2, mode='0'):
    """
    %DISTITAR calculates the Itakura distance between AR coefficients D=(AR1,AR2,MODE)
    %
    % Inputs: AR1,AR2     AR coefficient sets to be compared. Each row contains a set of coefficients.
    %                     AR1 and AR2 must have the same number of columns.
    %
    %         MODE        Character string selecting the following options:
    %                         'x'  Calculate the full distance matrix from every row of AR1 to every row of AR2
    %                         'd'  Calculate only the distance between corresponding rows of AR1 and AR2
    %                              The default is 'd' if AR1 and AR2 have the same number of rows otherwise 'x'.
    %                          'e'  Calculates exp(d) instead of d (quicker because no log is necessary)
    %           
    % Output: D           If MODE='d' then D is a column vector with the same number of rows as the shorter of AR1 and AR2.
    %                     If MODE='x' then D is a matrix with the same number of rows as AR1 and the same number of columns as AR2'.
    %
    % If ave() denotes the average over +ve and -ve frequency, the Itakura spectral distance is 
    %
    %                               log(ave(pf1/pf2)) - ave(log(pf1/pf2))
    %
    % The Itakura distance is gain-independent, i.e. distitpf(f*pf1,g*pf2) is independent of f and g.
    %
    % The Itakura distance may be expressed as log(ar2*toeplitz(lpcar2rr(ar1))*ar2') where the ar1 and ar2 polynomials
    % have first been normalised by dividing through by their 0'th order coefficients.

    % Since the power spectrum is the fourier transform of the autocorrelation, we can calculate
    % the average value of p1/p2 by taking the 0'th order term of the convolution of the autocorrelation
    % functions associated with p1 and 1/p2. Since 1/p2 corresponds to an FIR filter, this convolution is
    % a finite sum even though the autocorrelation function of p1 is infinite in extent.
    % The average value of log(pf1) is equal to log(ar1(1)^-2) where ar1(1) is the 0'th order AR coefficient.

    % The Itakura distance can also be calculated directly from the power spectra; providing np is large
    % enough, the values of d0 and d1 in the following will be very similar:
    %
    %         np=255; d0=distitar(ar1,ar2); d1=distitpf(lpcar2pf(ar1,np),lpcar2pf(ar2,np))
    %

    % Ref: A.H.Gray Jr and J.D.Markel, "Distance measures for speech processing", IEEE ASSP-24(5): 380-391, Oct 1976
    %      L. Rabiner abd B-H Juang, "Fundamentals of Speech Recognition", Section 4.5, Prentice-Hall 1993, ISBN 0-13-015157-2
    %      F. Itakura, "Minimum prediction residual principle applied to speech recognition", IEEE ASSP-23: 62-72, 1975

    %      Copyright (C) Mike Brookes 1997
    %      Version: $Id: distitar.m 713 2011-10-16 14:45:43Z dmb $
    %
    %   VOICEBOX is a MATLAB toolbox for speech processing.
    %   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   This program is free software; you can redistribute it and/or modify
    %   it under the terms of the GNU General Public License as published by
    %   the Free Software Foundation; either version 2 of the License, or
    %   (at your option) any later version.
    %
    %   This program is distributed in the hope that it will be useful,
    %   but WITHOUT ANY WARRANTY; without even the implied warranty of
    %   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    %   GNU General Public License for more details.
    %
    %   You can obtain a copy of the GNU General Public License from
    %   http://www.gnu.org/copyleft/gpl.html or by writing to
    %   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    def lpcar2ra(ar):
        nf,p1 = ar.size()
        ra = torch.zeros(nf,p1)
        for i in range(p1):
            ra[:,i] = torch.sum(ar[:,:p1-i] * ar[:,i:p1], 1)
        return ra

    ar1 = ar1[None,:] if ar1.ndim == 1 else ar1
    ar2 = ar2[None,:] if ar2.ndim == 1 else ar2
    assert ar1.size(1) == ar2.size(1)
    nf1, p1 = ar1.size()
    nf2 = ar2.size(0)
    m2 = lpcar2ra(ar2)
    m2[:,0] *= 0.5

    if (mode=='d') | ((mode!='x') & nf1==nf2):
        nx = min(nf1, nf2)
        d = 2 * torch.sum(lpcar2rr(ar1[:nx,:])*m2[:nx,:], 1) * ((ar1[:nx,0]/ar2[:nx,0])**2)
    else:
       d = 2 * lpcar2rr(ar1) * m2 * ((ar1[:,0] * ar2[:,0] ** (-1)) ** 2)

    if mode != 'e':
        d = torch.log(d)
    return d


def lpcar2rr(ar, p=None):
    def lpcar2rf(ar):
        nf,p1 = ar.size()
        if p1==1:
            rf = torch.ones(nf, 1)
        else:
            if torch.any(ar[:,0]!=1):
                ar /= ar[:, torch.zeros(p1, dtype=int)]
            rf = ar
            for j in range(p1-1, 1,-1):
                k = rf[:,j]
                k = k[None,:] if k.ndim == 1 else k
                d = (1-k**2)**(-1);
                wj = torch.zeros(j-1, dtype=int);
                rf[:,1:j] = (rf[:,1:j]-k[:,wj]*rf[:,torch.arange(j-1,0,-1)])*d[:,wj];
        return rf

    def lpcrf2rr(rf: torch.Tensor, p: int = None):
        """
        Convert reflection coefficients to autocorrelation coefficients.
        
        Args:
            rf (torch.Tensor): Reflection coefficients of shape (nf, n+1).
            p (int, optional): Number of rr coefficients to calculate (default=n).
        
        Returns:
            rr (torch.Tensor): Autocorrelation coefficients of shape (nf, p+1).
            ar (torch.Tensor): AR filter coefficients of shape (nf, n+1).
        """
        nf, p1 = rf.shape
        p0 = p1 - 1
        
        if p0 > 0:
            a = rf[:, 1]
            rr = torch.cat([torch.ones(nf, 1), -a.unsqueeze(1), torch.zeros(nf, p0 - 1)], dim=1)
            e = a**2 - 1
            
            for n in range(1, p0):
                k = rf[:, n + 1]
                a = a.unsqueeze(1) if a.ndim==1 else a
                rr[:,n+1] = k * e - torch.sum(rr[:, torch.arange(n,0,-1)] * a, dim=1)
                a = torch.cat([a + k.unsqueeze(1) * a.flip(dims=[1]), k.unsqueeze(1)], dim=1)
                e = e * (1 - k**2)

            ar = torch.cat([torch.ones(nf, 1), a], dim=1)
            r0 = torch.sum(rr * ar, dim=1).reciprocal()
            rr = rr * r0.unsqueeze(1)
            
            if p is not None and p < p0:
                rr = rr[:, :p + 1]
            elif p is not None and p > p0:
                rr = torch.cat([rr, torch.zeros(nf, p - p0)], dim=1)
                af = -ar[:, p1 - 1:0:-1]
                for i in range(p0, p):
                    rr_i1 = torch.sum(af * rr[:, i - p0 + 1:i + 1], dim=1)
                    rr = torch.cat([rr, rr_i1.unsqueeze(1)], dim=1)
        else:
            rr = torch.ones(nf, 1)
            ar = rr
        return rr, ar


    k = ar[:,0] ** (-2)
    if ar.size(1)==1:
        rr = k
    else:
        if p:
            rr = lpcrf2rr(lpcar2rf(ar), p) * k[:,torch.zeros(1,p+1)]
        else:
            rr = lpcrf2rr(lpcar2rf(ar))[0] * k.unsqueeze(1)
    return rr


def v_findpeaks(y, m='', w=None, x=None):
    y = y.float().flatten()
    ny = len(y)
    
    if 'v' in m:
        y = -y  # Invert if searching for valleys
    
    if not (x is None):
        x = x.float().flatten()
    
    dx = y[1:] - y[:-1]
    r = torch.where(dx > 0)[0] + 1  # Rising edge indices
    f = torch.where(dx < 0)[0] + 1 # Falling edge indices
    if len(r) == 0 or len(f) == 0:
        return torch.tensor([]), torch.tensor([])  # No peaks found
    
    dr = r.clone()
    dr[1:] = r[1:] - r[:-1]
    rc = torch.ones(ny, dtype=int)
    rc[r] = 1 - dr
    rc[0] = 0
    rs = torch.cumsum(rc, 0)    # time since the last rise

    df = f.clone()
    df[1:] = f[1:] - f[:-1]
    fc = torch.ones(ny, dtype=int)
    fc[f] = 1 - df
    fc[0] = 0
    fs = torch.cumsum(fc, 0)    # time since the last fall

    rp = torch.ones(ny, dtype=int) * -1
    rp[torch.cat([torch.tensor([0]), r])] = torch.cat([dr-1, ny-r[-1][None]-1])

    rq = torch.cumsum(rp, 0)  # time to the next rise
    fp = torch.ones(ny,dtype=int) * -1
    fp[torch.cat([torch.tensor([0]), f])] = torch.cat([df-1, ny -f[-1][None]-1])
    fq = torch.cumsum(fp, 0) # time to the next fall
    k = torch.where((rs < fs) & (fq<rq) & ((fq - rs) % 2 == 0))[0]    # the final term centres peaks within a plateau
    v = y[k]
    
    tmp = k.float()
    if 'q' in m:    # do quadratic interpolation
        if x:
            xm = x[k - 1] - x[k]
            xp = x[k + 1] - x[k]
            ym = y[k - 1] - y[k]
            yp = y[k + 1] - y[k]
            
            d = xm * xp * (xm - xp)
            b = 0.5 * (yp * xm**2 - ym * xp**2)
            a = xm * yp - xp * ym
            
            j = a > 0
            
            v[j] = y[k[j]] + (b[j]**2 / (a[j] * d[j]))
            tmp[j] = x[k[j]] + b[j] / a[j]                         # x-axis position of peak
            tmp[~j] = 0.5 * (x[k[~j] + fq[k[~j]]] + x[k[~j] - rs[k[~j]]])   # find the middle of the plateau

        else:
            b = 0.25 * (y[k+1] - y[k-1])
            a = y[k] - 2 * b - y[k-1]
            j = a > 0            # j=0 on a plateau
            v[j] = y[k[j]] + b[j]**2 / a[j]
            tmp[j] = k[j] + b[j] / a[j]
            tmp[~j] = 0.5 * k[~j] + (fq[k[~j]] - rs[k[~j]])    # add 0.5 to k if plateau has an even width
        k = tmp + 1
    else:
        k = x[k]
    
    # add first and last samples if requested
    if ny > 1:
        if 'f' in m and y[0] > y[1]:
            v = torch.cat((y[0], v))
            k = torch.cat((x[0], k)) if x else torch.cat((torch.ones(1), k))

        if 'l' in m and y[ny-1] < y[ny]:
            v = torch.cat((v, y[ny]))
            k = torch.cat((k, x[ny])) if x else torch.cat((k, ny))
    
        if 'm' in m:
            max_idx = torch.argmax(v)
            k = k[max_idx:max_idx+1]
            v = v[max_idx:max_idx+1]
        elif w is not None and w > 0:
            valid = torch.ones(len(k), dtype=torch.bool)
            for i in range(len(k) - 1):
                if valid[i] and valid[i + 1] and (k[i + 1] - k[i] <= w):
                    if v[i] >= v[i + 1]:
                        valid[i + 1] = False
                    else:
                        valid[i] = False
            k = k[valid]
            v = v[valid]
    
    if 'v' in m:
        v = -v
    
    return k, v



if __name__ == "__main__":
    if False:
        vec = torch.tensor([0.5828, 0.1555, 0.0659, 0.9627, 0.7891, 1.0191, 0.8021, 0.9337, 1.2774,
            0.6652, 0.2469, 0.7878, 1.0477, 0.8446, 0.3514, 0.2045, 0.6269, 0.8134,
            0.3893, 0.0667, 0.0727, 0.8494, 0.9231, 1.1648, 0.7300, 0.4666, 0.6589,
            0.4349, 0.8065, 0.7987, 0.3181, 0.6604, 0.8666, 1.1453, 0.7726, 0.9063,
            0.3332, 0.6760, 1.1205, 1.0775, 1.1448, 1.1907, 0.6672, 0.8823, 1.1622,
            1.1028, 1.0320, 0.4282, 0.6662, 1.2384, 0.6740, 0.5164, 0.4082, 0.7260,
            0.5392, 0.9719, 0.4180, 0.8444, 1.1877, 0.8488, 1.0906, 0.8853, 0.4757,
            0.6925, 1.1481, 0.6866, 0.3012, 0.1615, 0.1208, 0.8750, 0.9037, 1.0548,
            0.4164, 0.4729, 0.9301, 0.3204, 0.7745, 1.0680, 0.6909, 0.9062, 0.3540,
            0.7790, 0.7640, 0.3068, 0.8140, 1.1809, 0.9215, 0.7712, 1.1838, 0.4700,
            0.5017, 1.0036, 1.0230, 0.9579, 0.4505, 0.9709, 0.9884, 0.2803, 0.3104,
            0.6582, 0.3189, 0.5293, 0.4325, 0.8908, 1.1828, 1.0569, 1.0884, 1.0802,
            1.1001, 0.4115, 0.9149, 0.7085, 0.6449, 0.7778, 0.3562, 0.4321, 0.7873,
            0.3576, 0.6629, 0.8334, 0.8180, 0.3459, 1.0375, 0.8443, 0.9564, 0.7044,
            1.0270, 1.0423, 0.7254, 0.6990, 0.5834, 1.0301, 0.4865, 0.8758, 0.7471,
            0.8564, 0.3402, 0.1603, 0.5723, 0.5090, 0.5892, 0.7646, 1.2376, 1.1985,
            1.0682, 0.9348, 0.9846, 0.9919, 0.3230, 0.5222, 0.4669, 0.8245, 0.5602,
            0.9462, 0.8619, 0.2830, 0.0679, 0.2337, 0.7237, 0.8451, 0.7074, 0.7707,
            0.6635, 0.8395, 0.5148, 0.6486, 0.8791, 0.6315, 0.7330, 0.4579, 0.4154,
            0.9040, 0.3838, 0.2320, 0.6451, 0.9055, 0.6336, 0.3192, 0.9129, 0.5952,
            0.4966, 0.4688, 0.3329, 0.3772, 0.6293, 0.5530, 0.8698, 1.0336, 0.9268,
            0.8729, 0.5118, 1.0236, 0.6076, 0.9733, 1.0860, 0.4380, 0.1723, 0.8847,
            1.1701, 0.7571, 0.4500, 0.1477, 0.3908, 1.1088, 1.3307, 0.9212, 0.6443,
            0.3434, 0.5495, 0.6427, 0.5669, 0.1751, 0.9659, 1.2919, 1.0630, 0.3427,
            0.6302, 0.5932, 0.4882, 0.5931, 0.5578, 0.4152, 1.0133, 0.4814, 0.5088,
            0.4604, 0.3330, 0.1375, 0.7193, 1.0429, 0.9444, 1.0416, 0.3620, 0.3049,
            0.4035, 1.0365, 0.7757, 0.2473, 0.9898, 1.2477])
        lpcauto(vec, 10)
#    lpcar2rr(torch.tensor([[-0.0553, -0.3482,  0.7257,  0.5129,  0.0516,  0.3465,  2.0066, -0.2756, -0.6104, -0.6228, -1.7027]]))
#    distitar(torch.randn(11), torch.randn(11), 'e')
    
    signal = torch.tensor([0.902009179789033, 0.857449649918493, 0.877745763115494, 0.892742473447656, 0.877245754578477, 0.157539216623249, 0.156287880960755, 0.0355876386861848, 0.295410289476834, 0.0468164086485487, 0.102060671372019, 0.368965989255630, 0.181505909263061, 0.885677939232975, 0.599301266716556, 0.629693573388495, 0.803917408931684, 0.736749173474460, 0.0397427491013000, 0.0348085293088198, 0.752797537518358, 0.0798455720674639, 0.732635973361223, 0.946380690515890, 0.633688555940558, 0.385039981151952, 0.0220951355486199, 0.120310866181221, 0.191242438507035, 0.188801543068754, 0.336286318748004, 0.744156213748435, 0.111351307276400, 0.874953952810474, 0.890070549632688, 0.758524843387151, 0.370605155894207, 0.144177433468230, 0.647476214462572, 0.299180221571588, 0.955993004735538, 0.661012961840280, 0.885692473615400, 0.178818829569016, 0.726954828980342, 0.494291354093146, 0.741617850938414, 0.693302039032378, 0.270095017363227, 0.485059392825094, 0.0167923834439202, 0.602670753095068, 0.773653505182040, 0.638120890197460, 0.366763159511858, 0.374160791666788, 0.302015401589981, 0.244118361757156, 0.743650291567273, 0.0323325804327733, 0.796779049181593, 0.591109780992275, 0.914835477082632, 0.162588094537993, 0.272954885184294, 0.594342187803426, 0.743832432627816, 0.876678870175270, 0.996927285397191, 0.638121728212904, 0.128504700013970, 0.579758563015494, 0.906586754480664, 0.379262507206513, 0.570000505208623, 0.104390411406141, 0.695953311691443, 0.0550024419675101, 0.876811135184957, 0.999549269661403, 0.344669885696314, 0.493844142546554, 0.834669705345908, 0.596647549227296, 0.425609402770136, 0.581817790534000, 0.807779608105387, 0.530891155552800, 0.457607445107368, 0.600888534549647, 0.221222885121036, 0.961809785824140, 0.0739220422888305, 0.0309479918984584, 0.853055574358112, 0.741603342631630, 0.0449225911162017, 0.973401972055839, 0.375836619986697, 0.538795581946345, 0.0775069424090370, 0.258696119046371, 0.433044401736926, 0.167437537117021, 0.812567709328469, 0.906603959863360, 0.521027887183376, 0.124402050820987, 0.614239366172123, 0.192427966509229, 0.225241685775659, 0.244296683586666, 0.701533653297015, 0.830613289492705, 0.839517655905803, 0.887479037400514, 0.453611082331615, 0.920817561874756, 0.420667126182930, 0.414228757451924, 0.693005498480877, 0.647334837817861, 0.692822867556425, 0.283679579815263, 0.209285531381877, 0.499782740408831, 0.815017777013836, 0.908299806912029, 0.707288236983710, 0.961133751658944, 0.0396323778308312, 0.00242891908521337, 0.192740567937427, 0.107147133227754, 0.258560728478163, 0.452704708875755, 0.219396093070962, 0.585750856651409, 0.411947372294099, 0.417599453864577, 0.518581199263811, 0.361700706290302, 0.831564808987276, 0.598999250009971, 0.482218972477781, 0.245633888988540, 0.0562404267644926, 0.0491415690297414, 0.453980475848920, 0.674999233794652, 0.871351992434965, 0.0916743837448787, 0.606912016706691, 0.955749057136254, 0.646762104483008, 0.880352161228201, 0.983528169038712, 0.309482552108569, 0.628629058112563, 0.846841389758715, 0.407852616317842, 0.263559458422553, 0.658978328280686, 0.565231785473232, 0.232343615984342, 0.825190870957169, 0.248784308770198, 0.639006333938860, 0.474910129124576, 0.556343536176561, 0.351448567120949, 0.553578106729826, 0.912024525418144, 0.344375074426487, 0.922700122977673, 0.911716460231383, 0.260288801932781, 0.204315056087684, 0.795473765330989, 0.817561146393258, 0.630291848102722, 0.934083337759407, 0.650972687479680, 0.756516343476768, 0.481510135565163, 0.297206678379304, 0.576135128951585, 0.638975927107735, 0.729335575264733, 0.932366572462321, 0.156805839145418, 0.439960475699918, 0.359096925573261, 0.292587154133824, 0.0567572392668571, 0.0449614675001232, 0.859561978826997, 0.814935177749236, 0.0509029171931997, 0.372607923086495, 0.0306256957608234, 0.831832824670057, 0.727790761880902, 0.910231821913758, 0.700991809460803, 0.126656191142575, 0.0435943181984346, 0.128207494651727, 0.381728623852425, 0.883270866315957, 0.167978396149294, 0.120836601631621, 0.0199488767249236, 0.940855841532113, 0.403738583694766, 0.283365825053487, 0.506694765123206, 0.344088646555165, 0.236363190494259, 0.881140934648989, 0.697943968217679, 0.417107362258706, 0.0588833020952818, 0.545002295602310, 0.0213138338839750, 0.485253548849033, 0.0192089481904750, 0.800633432932323, 0.0769121971150409, 0.449157212412521, 0.803833986405562, 0.812479454715854, 0.0705704410918558, 0.798000471440036, 0.136978487578953, 0.391232750550029, 0.742999476253324, 0.941259179613167, 0.193354415472777, 0.396758584238710, 0.608140608907186, 0.922103523619876, 0.575322866550282, 0.0427441603351109, 0.443521203084316, 0.254625958379216, 0.954232039875055, 0.0773803265171177, 0.802726800285733, 0.563431821277259, 0.420152337425672, 0.400238787711914, 0.427147456220832, 0.0811092358704267, 0.385115598378648, 0.565703373540038, 0.0736978696498956, 0.378652557327386, 0.0703705976382515, 0.925192081888002, 0.228105978266907, 0.940766414949337, 0.403764008216319, 0.211593826458846, 0.584678115721820, 0.491684629855343, 0.981218349262454, 0.935205905852402, 0.807951147633083, 0.152709257524710, 0.283031883968284, 0.579638324697644, 0.335215479548609, 0.678002488421613, 0.0557463711488981, 0.285289876492141, 0.144830506318672, 0.287475397167327, 0.479422704804239, 0.475278586795728, 0.842038185557939, 0.984480152291859, 0.761285371930627, 0.586002234019310, 0.244379285050267, 0.628781977928054, 0.912769857348367, 0.930792006824367, 0.175204172789940, 0.552326140846925, 0.958044304520055, 0.128234899927329, 0.967776672626159, 0.0182332824517577, 0.282470700865027, 0.168396652165134, 0.362335310710506, 0.510675674745528, 0.0523491312826516, 0.372391771440439, 0.835328648563269, 0.644485240571894, 0.185756599921354, 0.140245469735218, 0.420560025409590, 0.957366533871680, 0.167242935744272, 0.935171220242417, 0.990025192094543, 0.520495439351741, 0.823089031527452, 0.395203309445679, 0.737739408100581, 0.226011208023241, 0.562079139421087, 0.765622965815698, 0.106088872014266, 0.322802506096501, 0.813006494647518, 0.708811181076159, 0.0900451763164634, 0.947750618419631, 0.962582970558386, 0.971648902950508, 0.270489713959826, 0.406935149242069, 0.127904778125129, 0.118724213646598, 0.327554615016101, 0.0692356005022332, 0.300158042473725, 0.481436853243026, 0.998042386076514, 0.106603486665915, 0.0184201900784935, 0.152907951549059, 0.571917218409074, 0.0308364632741607, 0.543150108585017, 0.541116327575533, 0.725964295519802, 0.953736373083477, 0.0171401658484033, 0.680018626183401, 0.396268225772336, 0.708749842569022, 0.873499085261649, 0.253833073585807, 0.901296014536058, 0.412926024519941, 0.920093646897932, 0.0173659836958305, 0.619734064903285, 0.752154998592706, 0.344414040326323, 0.765514808534426, 0.969715990259349, 0.238169901326236, 0.970537090624017, 0.212667418117534, 0.989977530295597, 0.942690336579539, 0.243125206166017, 0.134229704767345, 0.120566594000823, 0.754472614377275, 0.431225930227171, 0.864031291311622, 0.205659889745560, 0.685132199834811, 0.332037843713826, 0.754689287601755, 0.420609307211106, 0.178168576704746, 0.230044812274271, 0.0925206724628930, 0.139812135700818, 0.125909832141672, 0.783127804943179, 0.374680060434019, 0.597786412394450, 0.257454619666261, 0.0218257640958438, 0.911454993860656, 0.470036446036125, 0.0741159823126014, 0.545532807203954, 0.500987274966864, 0.296570365982951, 0.217175357222625, 0.0936392188228615, 0.925867411078883, 0.929421668157409, 0.591965515191132, 0.906971611054826, 0.855593190193936, 0.451062548463194, 0.995051245360120, 0.476478994551055, 0.979788034789043, 0.749411765054168, 0.110529414883354, 0.919883326232687, 0.618268981770698, 0.131767106144018, 0.190267132944580, 0.853192088715785, 0.521501346788466, 0.685645534678436, 0.473293671491970, 0.211703733849017, 0.0271263346693821, 0.311416487396719, 0.125185992312478, 0.148287590245424, 0.708494174698216, 0.404024041154871, 0.732322685115125, 0.0294326134814982, 0.907770570586338, 0.923990629673477, 0.827717821839196, 0.525935954271694, 0.501748927666033, 0.951485836523916, 0.286903766237051, 0.0845229310775242, 0.268736672714968, 0.335848383291941, 0.492274066213828, 0.335327256402237, 0.813090357292219, 0.131095372316170, 0.354826779409420, 0.490563292687267, 0.293658629907716, 0.367146062787293, 0.768845351679040, 0.399197284186901, 0.655363818658754, 0.766448131809319, 0.0986416150783442, 0.453714115953731, 0.441625770877606, 0.0411139741693831, 0.260018290460425, 0.0512623702275885, 0.946752639378858, 0.276009007410689, 0.941062186787285, 0.106660547733804, 0.993026441964096, 0.163252722360324, 0.318384187419039, 0.311179508877049, 0.0273837056000161, 0.762419544860343, 0.483788859620837, 0.453836558601393, 0.371195901874994, 0.0677148299891457, 0.123078184540142, 0.243568601435667, 0.368233104194634, 0.0416780019618825, 0.577278160071495, 0.549519158930995, 0.539945071299027, 0.232238908166894, 0.911325138454570, 0.354042980254358, 0.655043798018719, 0.632929110257966, 0.712691025492151, 0.341584706283263, 0.260443980697851, 0.185676356996808, 0.855939942030176, 0.206662844516314, 0.376239036659553, 0.301582338418767, 0.371673312652588, 0.0294935893242322, 0.0890773934266566, 0.122571778108006, 0.570935993373043, 0.00609360445721796, 0.100495602530315, 0.339729423424808, 0.388306520650410, 0.440720622008824, 0.0743961029798698, 0.446501554175616, 0.904708475424920, 0.891972490503782, 0.0385728138980795, 0.0191479303553825, 0.383662030300910, 0.297076176347438, 0.494905688296849, 0.268578600838695, 0.292868041838969, 0.452479202341218, 0.410688716609396, 0.426509387311117, 0.123837373238857, 0.785605525520462, 0.407021607332929, 0.831391334927907, 0.0410458457004099, 0.578604040835801, 0.991280128612868, 0.535594975087096, 0.476071574332275, 0.248476309360337, 0.273320128330826, 0.607433615834510, 0.594668142742809, 0.567673709991768, 0.860085867634661, 0.771335544702851, 0.131233180558464, 0.752830256788837, 0.255518987034766, 0.138207142819577, 0.167410944668102, 0.650381045512958, 0.549744349902788, 0.855475579279126, 0.463858734865807, 0.254743025196252, 0.490545008050392, 0.221729341380353, 0.139363618214790, 0.890195869275026, 0.561145778464440, 0.773899254154859, 0.750224332846226, 0.304661633310185, 0.0388238640547497, 0.149898095675492, 0.117703413177602, 0.00431050995790683, 0.774144233230024, 0.481150873540518, 0.219371710848083, 0.439428010421419, 0.0235452280272120, 0.603643167180409, 0.236233077509997, 0.891006788477140, 0.288073139516147, 0.142863609998161, 0.621432608776431, 0.625047154427755, 0.966189055762541, 0.0531961729471039, 0.271821240548085, 0.867525696580505, 0.852175558137918, 0.104447967570578, 0.870553663186548, 0.412798174184539, 0.395580778163553, 0.126235810055502, 0.0409544771593956, 0.107857223017966, 0.976000648235534, 0.336127569851047, 0.979410770495403, 0.248547732666452, 0.387316701576381, 0.430141584348472, 0.817198493036213, 0.607286517592843, 0.721590918208666, 0.621168371515182, 0.987947275221923, 0.122495320698168, 0.395765833729450, 0.0861198583281023, 0.411056624591203, 0.790255649224602, 0.822570158642709, 0.160375663393274, 0.619325533521107, 0.713861089444004, 0.0834969541246573, 0.698208597914411, 0.0461788592865406, 0.791181942086234, 0.857203276176444, 0.600870047985609, 0.480048077291840, 0.0747404854137760, 0.330878854913560, 0.00759929925225988, 0.439965806163396, 0.646057337480826, 0.590166346292590, 0.118819868029616, 0.719207929273137, 0.189117385538323, 0.759154209171390, 0.913214522475544, 0.808044954158004, 0.0571480266949620, 0.351302653748493, 0.665268679171387, 0.324490880443798, 0.413811516072220, 0.526930103467844, 0.757072986602427, 0.740754040413082, 0.403953566894323, 0.808399902621723, 0.202268809475871, 0.297014323967065, 0.912433327192783, 0.522672803159813, 0.301238553325982, 0.764851450565621, 0.641522517626095, 0.148926457607719, 0.774919025152975, 0.635060828690480, 0.905521557115379, 0.963645122186789, 0.343494386386706, 0.220295552271751, 0.649196599004296, 0.416609833074373, 0.444676907437189, 0.650256195610195, 0.568548511709625, 0.583731173235648, 0.453341925865745, 0.747583423723137, 0.966970439617405, 0.745547976962830, 0.263493610583575, 0.286375031309663, 0.544061509485368, 0.175725396683641, 0.432009584311002, 0.658283705791754, 0.624778500940352, 0.336494527752326, 0.547481408141862, 0.957102579040529, 0.885520034097712, 0.260938848065239, 0.812864680060181, 0.0580596250137596, 0.0203663376579579, 0.834875557097982, 0.816274803151954, 0.0341781169643275, 0.674634074393312, 0.924659638307041, 0.922042791736362, 0.650225143305336, 0.184115247106255, 0.0958274716351915, 0.741650283060460, 0.0546294833921486, 0.0217529142147298, 0.0608284057650957, 0.468470929732757, 0.0337700218514384, 0.532830687583506, 0.858905703036378, 0.221008004134506, 0.565965195492701, 0.0441236939590249, 0.961573428612896, 0.705961054046688, 0.0872386640939127, 0.0724918583665276, 0.426898998297564, 0.349212713389637, 0.478341925027595, 0.0812400286737953, 0.0336294885453109, 0.275928227893847, 0.692347957203507, 0.874824865314447, 0.344941750177836, 0.262175356390432, 0.506806631315011, 0.351334018252692, 0.505927600616046, 0.158424553436491, 0.635389707034827, 0.00406185451786667, 0.743750705652948, 0.386866122073472, 0.0417351757146074, 0.378749921960199, 0.387466086159764, 0.780754499999310, 0.657960782221536, 0.519799680602724, 0.581303298191537, 0.241731682487381, 0.825692740238403, 0.202602429596367, 0.752880784184643, 0.719265655339237, 0.322494605393209, 0.916472789299027, 0.0563992589886198, 0.947929023798585, 0.250756812244631, 0.572188573387072, 0.911747158961009, 0.179586367127237, 0.930245677897527, 0.801368797028642, 0.573255243385268, 0.746903832945480, 0.0807083036894496, 0.231919119618704, 0.228400341368125, 0.0209813235483984, 0.167344685673257, 0.00581709680525044, 0.554625963247733, 0.0332241389273414, 0.869351102179318, 0.779196525260889, 0.00336986931952366, 0.0313359894995122, 0.288324075739082, 0.316854416584293, 0.937663242960437, 0.907815765026449, 0.631184899480714, 0.154859327079241, 0.138448332634541, 0.513591937909179, 0.893500302153128, 0.675732013034023, 0.258469639017058, 0.862408210835207, 0.664844267768580, 0.858021973415105, 0.948484400011509, 0.266232571663563, 0.307708820329816, 0.542005294011343, 0.682641644962506, 0.424954821462961, 0.369830746709438, 0.303659796827579, 0.972460208269960, 0.586320897350485, 0.420587187672049, 0.860507140218360, 0.0605331406561002, 0.682419014059445, 0.533831312085226, 0.789925420198534, 0.322109382241374, 0.764223140784458, 0.953810599205749, 0.890321858228180, 0.467839978552316, 0.842027501234104, 0.0227509328320699, 0.271995947269178, 0.0246236500953335, 0.928682462823008, 0.996620360461477, 0.911469160536051, 0.910758828809596, 0.568138514623280, 0.0904693097374417, 0.794978053474609, 0.158978252555318, 0.690978545859684, 0.306232906770187, 0.375331877621115, 0.709955655211423, 0.272448212704643, 0.686147129600441, 0.667620959580175, 0.300925126741733, 0.0162633601429097, 0.287595755155619, 0.926453572147091, 0.646201419847575, 0.688129110722537, 0.917936220444640, 0.534110189734972, 0.511597380038961, 0.711049331519633, 0.151861105726329, 0.718254977638095, 0.177178599420045, 0.672279880004321, 0.106305252447427, 0.801251403927116, 0.910894279399749, 0.614129741398296, 0.954051425045267, 0.892411996988366, 0.328543860231569, 0.813991812555092, 0.132109027002903, 0.253461951307272, 0.447349847561302, 0.0526510055465780, 0.0482358826957525, 0.492680785174112, 0.576990562088458, 0.743509313020217, 0.166232645761074, 0.189013134435012, 0.0390196471169793, 0.457439809484564, 0.380070774717204, 0.833305668128222, 0.479993697795805, 0.572793071696908, 0.892049655564182, 0.867604014113090, 0.766324223864600, 0.570735736968777, 0.0177932530502698, 0.508853712165056, 0.666427878228609, 0.0253865478979681, 0.733853347847352, 0.886575693051682, 0.666562380326770, 0.889686664420915, 0.461083823670775, 0.161418669289312, 0.640722323627741, 0.171469650181642, 0.577006972846817, 0.0268272539530239, 0.566857369582035, 0.788942946725889, 0.399581076007310, 0.815454516160776, 0.578806797115959, 0.849646082877172, 0.555739204603788, 0.127054006836184, 0.640658280294968, 0.127609489457398, 0.0952709344329532, 0.615251123551125, 0.560858018549798, 0.100130759084160, 0.860155312718600, 0.661403549385742, 0.406537565357070, 0.288441686783680, 0.556851447093619, 0.371111821580571, 0.954766558441460, 0.594116212986737, 0.820320757582310, 0.921820072492265, 0.305457213777172, 0.666017953533942, 0.933516558551661, 0.0969933585889722, 0.881984083190056, 0.876759774428236, 0.0588199171457253, 0.0562532388337153, 0.0505811886456189, 0.222463771384000, 0.910057942143447, 0.958663896809541, 0.595810869838331, 0.413196171461557, 0.0672085954703464, 0.942453530603068, 0.921333131413558, 0.747163401911716, 0.731083823745944, 0.802332667215361, 0.514568146075506, 0.594186937163955, 0.248351027672830, 0.577734169733868, 0.290914013947906, 0.666704478959694, 0.985982022235759, 0.597974444004296, 0.439042797279193, 0.692077528227061, 0.922508252282822, 0.645478081425729, 0.476563329325597, 0.0905240342520427, 0.707116207702985, 0.930295456244946, 0.915224413475178, 0.211508589170455, 0.159956103258252, 0.434296703065965, 0.566578784719599, 0.283266817656958, 0.442858615773550, 0.938760212397327, 0.00953103179454562, 0.136487337895532, 0.645267714689690, 0.638850956625339, 0.750146619080334, 0.964775068396771, 0.569450526439654, 0.833975659351619, 0.561198148640256, 0.175193555600016, 0.389203653508421, 0.555888925239606, 0.533676222450874, 0.110771304969843, 0.268297883973440, 0.571374341342982, 0.917811460587834, 0.0987143065064207, 0.187281624336678, 0.597641476055817, 0.741089227865776, 0.254118109298116, 0.423937139051219, 0.273450864972187, 0.676034608047999, 0.772793180897711, 0.0964675214588511, 0.502570135261928, 0.691983911307480, 0.100799779571109, 0.268517671303857, 0.805426708563206, 0.733785630138861, 0.687434171868995, 0.310902864037329, 0.428245772550059, 0.713893028992675, 0.284319461897049, 0.870203929250021, 0.246887772331371, 0.743679491359757, 0.0646103207021664, 0.599709591086680, 0.277230315004554, 0.0124405123045114, 0.247290258512336, 0.865367630397176, 0.257829898410626, 0.852283537149405, 0.722883016559493, 0.598240475805238, 0.512296331033200, 0.0655366615530242, 0.221559634442891, 0.752189360953619, 0.532634746335622, 0.784460187768816, 0.534107317226228, 0.799083452539956, 0.0742722819116667, 0.922268485225477, 0.515507503715463, 0.372964961226051, 0.756848149366405, 0.898176058978502, 0.523711251579790, 0.249116758610378, 0.297417093977592, 0.844169566084752, 0.550726790956814, 0.763496462100689, 0.285355237233653, 0.860812528172914, 0.215340105685542, 0.423172227780775, 0.527806106025651, 0.427711166857355, 0.614845227140402, 0.0952616613954544, 0.185287485710505, 0.0829702140126092, 0.0749731763003131, 0.557422534245224, 0.667819786344377, 0.521783161774332, 0.373407269884763, 0.486882306491630, 0.770733081256417, 0.354613566619759, 0.746481740394125, 0.305921796588920, 0.130277579903150, 0.367885994709369, 0.0373660834564118, 0.796561207775974, 0.216706581927159, 0.179211210782796, 0.856516152082066, 0.469387025810014, 0.0406896442196901, 0.783264251372380, 0.977041699339204, 0.0356508626041904, 0.444066170846211, 0.650791900770142, 0.915837246247873, 0.620091769079586, 0.399018914238702, 0.902616394640571, 0.215104342525860, 0.907429976465391, 0.506566205757690, 0.179930099944538, 0.0585777256949440, 0.852028335721361, 0.578820781515225, 0.886964654862463, 0.322725043490731, 0.911422074890074, 0.727872011120242, 0.559612984571835, 0.518555254750624, 0.365080444886158, 0.388100995580500, 0.523165911040278, 0.323350968985978, 0.813884866482664, 0.0891789894920950, 0.466255276332188, 0.902596845886773, 0.465788865797507, 0.890240778779375, 0.204435803169980, 0.649350809579490, 0.601473294880583, 0.964931125057361, 0.553541073852539, 0.462470892863768, 0.121741031968208, 0.275252268549276, 0.122053648817994, 0.966479126320934, 0.472080095857187, 0.324894047071195, 0.00414324055696413, 0.945996171017745, 0.683670664365147, 0.183528496960106, 0.961910937600611, 0.299262727627282, 0.981727200191125, 0.909148835212793, 0.714010806289901, 0.666302218138895, 0.786742787224698, 0.805330778227449, 0.539616862671517, 0.849228849095167, 0.476658004356420, 0.304012145824231, 0.325817050838687, 0.255847830220836, 0.513834582118065, 0.706532258109939, 0.993943454020144, 0.256124798700520, 0.611177785363453, 0.173132978759554, 0.238768113636916, 0.881575790304413, 0.226345364765027, 0.787033555445779, 0.761619087149148, 0.616560336052824, 0.187915466787862, 0.481817303430996, 0.498022002797683, 0.164658607339409, 0.00333997289657728, 0.526872722800437, 0.330336580808116, 0.238876506581845, 0.696865089695026, 0.534198089556253, 0.0229983749939187, 0.956009196337294, 0.503153163844744, 0.142654895730775, 0.239722371337688, 0.912624972815205, 0.205271318492281, 0.264422925128537, 0.928473171817544, 0.220212457781670, 0.0369907674890889, 0.632227525413753, 0.268337736789682, 0.838795458662828, 0.635954765808034, 0.146169575775330, 0.640924631670122, 0.862558179317782, 0.631560836578981, 0.305407254704058, 0.891427590524593, 0.867338848245795, 0.858469176007830, 0.987134299801114, 0.246665541843413, 0.173672009590321, 0.932908080837661, 0.151772918556492, 0.0356128093331956, 0.172499419868052, 0.828161851358829, 0.255869277148244, 0.217634374211302, 0.822764313145968, 0.723638371196358, 0.662511420054018, 0.202654842972586, 0.0705463192973228, 0.944277133430553, 0.602693652510915, 0.935225688210900, 0.398461797821548, 0.969114138190855, 0.0634451761213043, 0.506650871556580, 0.612851385785104, 0.337621415120122, 0.700747452119724, 0.909816368310075, 0.271372204478763, 0.704073483270247, 0.928715941304830, 0.497274299951722, 0.0962538304477525, 0.733785738636717, 0.416347288961749, 0.0856303570335012, 0.0332138647055387, 0.488626414136107, 0.957903592446415, 0.743899294149054, 0.400290403366713, 0.0214058920325796, 0.561788045690476, 0.616987220839504, 0.541087160393456, 0.495570795258479, 0.919907147826719, 0.470093806023199, 0.439676705067586, 0.154027640792507, 0.429079973535708, 0.978344567466805, 0.950409523104309, 0.454373787941690, 0.0369849997322944, 0.962980756225965, 0.519925724763064, 0.535075675954443, 0.349408377971135, 0.833098280621675, 0.717711050753576, 0.0272910591902497, 0.290506890462008, 0.702205187963534, 0.699090663886114, 0.909258979118747, 0.577243546120286, 0.864315087316640, 0.915753400272056, 0.213722109436058, 0.681399480366033, 0.143113246641640, 0.462738275464301, 0.909460963690416, 0.798441910528251, 0.719684745334300, 0.545484298088812, 0.673967847845076, 0.145963111559603, 0.0101105976140031, 0.461433067804146, 0.262753900548819, 0.469355380590357, 0.193139261569515, 0.394403855426966, 0.871508260194195, 0.896479576244688, 0.355695894840571, 0.879039389516314, 0.531069702207970, 0.0876272888965782, 0.826171491547228, 0.0891146426957613, 0.595693570486773, 0.875541352936600, 0.679784279545128, 0.688548366106875, 0.487527680004526, 0.141267103656650, 0.362379210967288, 0.465897892048472, 0.787225754897473, 0.487495671835901, 0.638512170776324, 0.156065853498785, 0.935266862858513, 0.00337413006983578, 0.0618855735053151, 0.390228938469510, 0.608508913780345, 0.637629261537963, 0.924510398554809, 0.466924946358892, 0.260060244393170, 0.159801017170561, 0.640540180860743, 0.859516208941552, 0.0801465017423014, 0.403167523853527, 0.571176200225743, 0.226890362420408, 0.673166496481169, 0.732648256991095, 0.806664026332060, 0.623065954395349, 0.202015835947028, 0.823995810024170, 0.772181215976551, 0.600888982752864, 0.934769355920321, 0.490985366230808, 0.208862925807873, 0.571324773747848, 0.0749804300927540, 0.0104394366885155, 0.0239978753561891, 0.133077968774761, 0.437779576500499, 0.996882255610456, 0.719973121040105, 0.998541662060513, 0.0221462725515147, 0.957504887081753, 0.484919062303162, 0.608617831870754, 0.679961975040096, 0.973292511311736, 0.584518651400675, 0.968844366770315, 0.579481144844334, 0.271180461507889, 0.406699054772426, 0.563649226356041, 0.247515329319496, 0.0953609588887560, 0.0954442699596840, 0.337221474896835, 0.277634775529961, 0.446984021357111, 0.693374183176783, 0.133687647906552, 0.476655572624605, 0.791923826642163, 0.0618253961936411, 0.0180909319650067, 0.573201629210076, 0.652187123510770, 0.778701770617668, 0.927625322285883, 0.409244295425234, 0.251299739524502, 0.835439941962507, 0.880117771592748, 0.330759350694797, 0.0564552368802653, 0.392457847423438, 0.297883892057792, 0.217051861164894, 0.494709007350811, 0.517295722617896, 0.780527184841327, 0.304485592496799, 0.406307255547413, 0.290851624081954, 0.0324899456333253, 0.875437541374476, 0.440852858671448, 0.937973146305091, 0.468749055772896, 0.651532497214833, 0.697790162122535, 0.0703282730981462, 0.700475312613738, 0.308380552739526, 0.991839960039810, 0.878655600408595, 0.761185828045384, 0.483824132861068, 0.388257401390740, 0.286493894779582, 0.243004513484238, 0.457040762865441, 0.431433427005834, 0.387362817203419, 0.409938582477554, 0.0807410294469808, 0.112919878144427, 0.107629059535187, 0.502902131369572, 0.505505938558785, 0.996488295196870, 0.653927872794009, 0.229317947858823, 0.396106849623076, 0.234797417385602, 0.0720823992517505, 0.900898648073367, 0.590538253246551, 0.475304899218811, 0.667305222484094, 0.665487014967114, 0.775024979746692, 0.475457940483864, 0.451075474816991, 0.106651298314717, 0.0351953490992319, 0.0668717253079356, 0.486206255029126, 0.281932894270035, 0.478261968961981, 0.665401490547795, 0.211544157206745, 0.954994587467716, 0.641985055577558, 0.716902142935964, 0.801484733844874, 0.545812328566623, 0.153410602491986, 0.532400227640497, 0.0909862914071590, 0.674884134779967, 0.284591455826545, 0.336758247480253, 0.392343919664016, 0.619516599236488, 0.562694609711269, 0.883818317486446, 0.113127257235823, 0.0160555946483956, 0.385346935737643, 0.688438933318894, 0.648394689261057, 0.319187671440301, 0.517129962116029, 0.912057575857568, 0.121301101697799, 0.324986300938097, 0.928224637832109, 0.255102598637714, 0.717980370884329, 0.501543299848027, 0.815108311380902, 0.740362834544025, 0.321177966768924, 0.540122997939341, 0.392079952406418, 0.951723282855147, 0.697793220172706, 0.404544863823983, 0.410950987314803, 0.449448694775019, 0.954978930937661, 0.600233256897472, 0.154777365214337, 0.269989937040735, 0.223060570257244, 0.974963648021044, 0.119125547291389, 0.130421099034893, 0.180299831731766, 0.285140554517355, 0.0698096518916462, 0.954494018325097, 0.327524182159664, 0.934372613486256, 0.918312116847896, 0.719781685565002, 0.554470459606452, 0.614379419882338, 0.511145021347751, 0.0301127513960687, 0.923497711913674, 0.849112322371445, 0.0988354120976046, 0.611347419756223, 0.214287860207259, 0.595443698550536, 0.741744676039944, 0.594309488040603, 0.821413959009877, 0.704064895200178, 0.0242692623368519, 0.178140777802288, 0.895020031275000, 0.787481642164826, 0.211118838975538, 0.272569769247711, 0.787317774747775, 0.390739290063677, 0.944247297658050, 0.591891238542496, 0.545156838041856, 0.839635825076825, 0.0690887221146396, 0.574029706917016, 0.700908979526840, 0.634972289508269, 0.217453651155332, 0.551525410458929, 0.816847904090836, 0.238211112396908, 0.523229860710028, 0.791063535095643, 0.149799133228904, 0.505931495246794, 0.754795051024504, 0.793064462034549, 0.925042947217229, 0.821996892248024, 0.798828811183295, 0.998901069449087, 0.771804319326477, 0.0632072116316004, 0.0548469790951585, 0.293934885073060, 0.816555124479486, 0.545915879119805, 0.334919287715122, 0.975539878967631, 0.577293260003949, 0.579617098964916, 0.432376939008802, 0.194010928720934, 0.415341825800883, 0.303346113951632, 0.807105643172537, 0.350890143576588, 0.539750857159985, 0.934380996845425, 0.0446431856084658, 0.829372165747720, 0.340785332562361, 0.835103707891518, 0.780568605574363, 0.804677021878349, 0.148508367855958, 0.252255113897827, 0.616282139307474, 0.660930126379468, 0.592141059888696, 0.776266230773379, 0.0535238602579086, 0.796379268055215, 0.520334047767841, 0.625247494190599, 0.0696434506607899, 0.985966729763871, 0.0342035191234787, 0.0851606593734150, 0.544021678675166, 0.395210973236571, 0.766996976833041, 0.793419423944433, 0.814009975311062, 0.147562859721123, 0.980807277982547, 0.309035366571600, 0.873489666750080, 0.699561594453495, 0.512097647062512, 0.580294448931371, 0.715276625787071, 0.359491470461607, 0.920754146405014, 0.493925700811349, 0.199645176745432, 0.0190549562334760, 0.579227443825498, 0.274715405279902, 0.331864435849025, 0.414228022164123, 0.637178834274798, 0.621419601387161, 0.224128549261095, 0.376436948523142, 0.296692342953643, 0.811850006186711, 0.741433806425687, 0.817377912708006, 0.346859453596768, 0.220755804457651, 0.754018472677165, 0.840410554827926, 0.186487512139446, 0.218010741349360, 0.432071524128335, 0.431218655090067, 0.143466085112315, 0.683271341840964, 0.401092175547482, 0.965122178442258, 0.0888068208515961, 0.105764863315801, 0.191053843328913, 0.848816469267908, 0.639001778539516, 0.970792267314407, 0.349192807255925, 0.756428702720984, 0.870599980642354, 0.406433996817725, 0.404399288228119, 0.720061710351491, 0.976918921573116, 0.783545897902607, 0.581195633079232, 0.103013954825107, 0.522776451788199, 0.134264783160118, 0.770620808818079, 0.614522379426795, 0.393581721637556, 0.886369036335903, 0.378179759827406, 0.697338207145749, 0.498195759230357, 0.652773282518988, 0.848961652823128, 0.184219081488682, 0.629436513827010, 0.0730062057365410, 0.745544289960073, 0.802214749881686, 0.913398090487757, 0.866044662249448, 0.584452024774763, 0.0436523372229861, 0.532997804431872, 0.341031651381389, 0.945394720343237, 0.809048348692904, 0.231293451334051, 0.344238963099934, 0.487551301679215, 0.725243879662631, 0.463784731925247, 0.468594640315211, 0.680356667025199, 0.962673577834282, 0.187167503029483, 0.947587520456619, 0.739175741236602, 0.190863446554048, 0.939832566738028, 0.617082854052840, 0.0746210112713538, 0.0190999376686334, 0.108808975660537, 0.676529503718514, 0.935594265712407, 0.942742916383958, 0.507171251373251, 0.232715103279943, 0.878994285017821, 0.687627220228940, 0.658280555267701, 0.254042109902179, 0.461770310436855, 0.875618277460207, 0.167512393359483, 0.0325806969740291, 0.655776665121549, 0.903450621332904, 0.914791229026784, 0.493096866743431, 0.631549151303902, 0.947968910136151, 0.342934513273344, 0.225894053893771, 0.0833365745213834, 0.00811716537984664, 0.368607793865863, 0.652881180021635, 0.760335993742284, 0.998277225791299, 0.660337818751145, 0.461151869567180, 0.825663348157992, 0.973382130061684, 0.136105200589707, 0.460929509834195, 0.184942056721233, 0.970107066267215, 0.0570738681382482, 0.140114461772214, 0.0530268923097587, 0.740898095525236, 0.214170508655938, 0.243708242841471, 0.656666689136133, 0.00530988509604391, 0.989714621738085, 0.900571526843981, 0.535104360852452, 0.567819978454158, 0.641960473568125, 0.315012180540646, 0.514448426518322, 0.156332543127019, 0.184113248218095, 0.892066253920420, 0.555637477793676, 0.205632368889712, 0.796653362542720, 0.411736061391268, 0.558009719231763, 0.391500589413957, 0.597442851610202, 0.347459216809887, 0.916925030296183, 0.152822097534285, 0.801622063827527, 0.0655943266116928, 0.167869645864372, 0.775140816011310, 0.298162571369961, 0.728805992389838, 0.341169408362424, 0.459470782675546, 0.775872441459016, 0.609833807927448, 0.632999319613307, 0.0452411694527557, 0.697899055274699, 0.780323434621162, 0.477508271684331, 0.272493710450044, 0.181252707639444, 0.551064653221325, 0.789392092566139, 0.362195692500510, 0.207769843346935, 0.410836571848061, 0.778358287063334, 0.136192795803946, 0.869292894740801, 0.311743822876833, 0.768616384935376, 0.322687474861486, 0.160393859330797, 0.288053755165877, 0.188591827241362, 0.382550164354357, 0.458232159280107, 0.759493383293433, 0.940811960791694, 0.0261716929373115, 0.705262677507106, 0.00812599580869788, 0.961672179586050, 0.877253672667710, 0.826230830929768, 0.526097595038276, 0.658992787532069, 0.174383673408918, 0.412355477169225, 0.777844262402374, 0.320681077452227, 0.905562631056607, 0.203720087633085, 0.782259926588671, 0.795422678359720, 0.438422515532861, 0.0533992516135863, 0.0302698119895136, 0.951449499306147, 0.146214814757796, 0.694393314027902, 0.849799788513946, 0.680943846572585, 0.0217967761735102, 0.751454067417635, 0.893598104352329, 0.463409051024121, 0.716911129759880, 0.384721161789319, 0.0396016899133752, 0.917770168912550, 0.889469238524446, 0.262510773404725, 0.426318866717944, 0.0320728804985105, 0.105287116886867, 0.849309502142030, 0.974113065783969, 0.0102997376047518, 0.339382171308140, 0.705811789387468, 0.894050052840485, 0.732575642484519, 0.998570946332062, 0.173417850862611, 0.859064022119227, 0.621147901250123, 0.571007374340309, 0.954526395046550, 0.529953210057687, 0.532211064923099, 0.267585569394189, 0.755888986458908, 0.484699464503938, 0.756684895004418, 0.503770671032031, 0.645634695012856, 0.0247351592374596, 0.951294456846883, 0.357689049600523, 0.934030570339129, 0.653736534478240, 0.510507231349565, 0.761939814252758, 0.513127752031007, 0.567871099660631, 0.855470317155833, 0.772173374946471, 0.132237242577549, 0.454508024532270, 0.996564961663445, 0.940535615497928, 0.507385290857436, 0.561077847248289, 0.738785062661710, 0.462620984433571, 0.789527363020529, 0.265767542097456, 0.274437532947533, 0.951812804746640, 0.798344448953478, 0.0719319003945604, 0.381208158628879, 0.685682345565382, 0.700307669149310, 0.654738003377331, 0.0809482268435837, 0.290822457815510, 0.268427401082689, 0.379196511622666, 0.532762569504827, 0.180564326741716, 0.146690998077345, 0.290379040768665, 0.196288304889383, 0.592416834436414, 0.876066889803569, 0.344029401520705, 0.868282490020277, 0.745009734854917, 0.133688798243210, 0.716290433037331, 0.721683034582533, 0.447561235860577, 0.955890472286763, 0.141689569533651, 0.681105187429444, 0.357479872132484, 0.537818383328480, 0.206873140973020, 0.761894176136185, 0.00519375417029566, 0.874119559846713, 0.805949196754096, 0.823996604493778, 0.953011878220868, 0.335543388902552, 0.146814027069480, 0.853040819059046, 0.223400203298820, 0.839098982868905, 0.397760850131843, 0.235018755727724, 0.845232289035112, 0.851647670373618, 0.788447242059125, 0.263099205176062, 0.667111269823311, 0.247664065689713, 0.395154970905836, 0.457894215279718, 0.661997419268891, 0.283009457275576, 0.335800070291014, 0.355630605972652, 0.148290168508597, 0.754663325586713, 0.227691147692654, 0.100811086343667, 0.406984447910267, 0.610865230204408, 0.457536655068514, 0.255269877108299, 0.926292287100675, 0.542524159938723, 0.925120417605366, 0.920677090699961, 0.161443864954452, 0.613588057313364, 0.496669944544543, 0.249301093962130, 0.0583162357196400, 0.896736307993068, 0.431865498982512, 0.415495061840682, 0.405055179243034, 0.719290392071844, 0.320667199398132, 0.228860665288584, 0.314512621448852, 0.403985945302329, 0.169054854009786, 0.580611247775640, 0.0153666156662847, 0.357193565501887, 0.293788718011694, 0.827337078886027, 0.434426870586880, 0.387384527748627, 0.0618735958561687, 0.986738224625924, 0.670119620133540, 0.942418797459675, 0.774582036686533, 0.433402604578237, 0.498753015899121, 0.387622625581322, 0.143594866968921, 0.813175065188728, 0.784238615593936, 0.811163368506479, 0.574212763374683, 0.681502829024468, 0.357866340968659, 0.858048420636625, 0.457867345908557, 0.494064040471442, 0.789627811794001, 0.212487970999362, 0.347470682677814, 0.840274316192818, 0.385372820997361, 0.706851293027159, 0.472175982672588, 0.248285508213315, 0.604238112591811, 0.401129919388795, 0.742227257374185, 0.245027917466076, 0.749919393645994, 0.0109786008643363, 0.919578618554074, 0.961257817539976, 0.335075313643315, 0.782791285861568, 0.457756996372522, 0.112731138592039, 0.818557706331381, 0.198787167086759, 0.340289510642708, 0.161558973731012, 0.455961433888531, 0.285679252840478, 0.107770724378252, 0.395187164951989, 0.214476922314966, 0.106704548366523, 0.512046866510653, 0.891528204710678, 0.758599263158441, 0.782102810377518, 0.322388673023457, 0.876938234329390, 0.382469822789647, 0.336676491160082, 0.567532195435175, 0.849160346264891, 0.0660392926854873, 0.289887849838915, 0.384062706268051, 0.615287977066830, 0.557363503359153, 0.0364526089053425, 0.847822563050743, 0.471662687291774, 0.582077688467291, 0.561018791943536, 0.299337114314045, 0.602784431920052, 0.711460545905730, 0.0371983800290949, 0.588004099846756, 0.380917548151063, 0.211617070995011, 0.574307147178044, 0.391323053993246, 0.682150591829648, 0.0565257138252711, 0.308877682870720, 0.0561429322787286, 0.861293939466177, 0.364578538672096, 0.340048293509225, 0.0722164813152104, 0.102138511275985, 0.467689438358354, 0.881568931645432, 0.829029330808678, 0.342486601716722, 0.789811376827660, 0.262295595360756, 0.681546174813926, 0.979868132099842, 0.182857653854273, 0.701231880821244, 0.124467959309610, 0.963209979817285, 0.586554403371440, 0.716774644907597, 0.121978220912373, 0.484261699855706, 0.529221245518901, 0.374795923909195, 0.417489557388511, 0.770286524776386, 0.247437976156220, 0.498126614400758, 0.612120796854619, 0.183312144762481, 0.363429476726910, 0.850364397349446, 0.0818138319992320, 0.281079654875688, 0.317198836526924, 0.814274452603058, 0.703053610448434, 0.306681125364248, 0.584291200633655, 0.943092244009268, 0.712411623614839, 0.922905960734781, 0.0720852232834217, 0.367121534204545, 0.605368318199108, 0.782687815664646, 0.801331317497223, 0.123924220994347, 0.475405630851030, 0.211092739394820, 0.606002045011485, 0.585326962598841, 0.778750842765555, 0.548924112654356, 0.0777547175290193, 0.217319785180850, 0.291051102649413, 0.746104149667072, 0.161085502780684, 0.264220048917010, 0.262500620898859, 0.735176242668518, 0.0447898974154554, 0.703009537625969, 0.00247250653095243, 0.530821053976008, 0.140625193601345, 0.0387515936034518], dtype=torch.float32)

    fxrapt(signal, 8000, 'g')

#    fxrapt(torch.arange(10000, dtype=torch.float32), 8000, 'g')
    print(v_findpeaks(torch.tensor([1,-1,3,6,-8,0, 11, -2, 1,1,3,-6]), 'q'))   

    if False:
        x = torch.tensor([ 0.8059, -1.3887, -0.3314, -1.2292, -1.5568,  0.1364,  1.8636, -0.0174, 1.0607,  1.4072])
        y = torch.tensor([[-2.2671, -0.7761,  1.1568, -0.0453],
            [-2.1536,  0.6736,  0.3871,  1.5711],
            [-0.3061, -1.1705, -1.3570,  0.3299],
            [-0.9602, -0.1271,  0.1808,  0.0372],
            [ 0.0668, -0.2001,  2.0095, -0.5304],
            [-0.0354, -0.6978, -0.4232, -0.6803],
            [ 1.2171,  1.0186, -0.3193, -1.0853],
            [-0.6021,  0.4268, -0.9312, -0.0758],
            [ 1.4515, -0.9186,  0.3999, -0.2904],
            [ 0.1442, -0.9291, -0.1663,  0.7734],
            [ 0.5064,  0.8958, -0.7381,  0.8086]])
        normxcor(x,y)
        y = torch.randn(21)
        import ipdb ; ipdb.set_trace()
        print(normxcor(x,y))
