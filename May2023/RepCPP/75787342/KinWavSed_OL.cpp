#include "KinWavSed_OL.h"
#include "text.h"


KinWavSed_OL::KinWavSed_OL(void) : m_CellWidth(-1), m_nCells(-1), m_TimeStep(NODATA_VALUE), m_nLayers(-1),
m_routingLayers(NULL),
m_Slope(NULL), m_DETOverland(NULL), m_USLE_K(NULL), m_Ctrans(NULL), m_Qkin(NULL),
m_FlowWidth(NULL), m_DETSplash(NULL),
m_eco1(NODATA_VALUE), m_eco2(NODATA_VALUE), m_V(NULL), m_Qsn(NULL), m_Vol(NULL),
m_SedDep(NULL),
m_Sed_kg(NULL), m_SedToChannel(NULL),
m_ManningN(NULL), m_whtoCh(NULL), m_USLE_C(NULL), m_Ccoe(NODATA_VALUE), m_WH(NULL),
m_streamLink(NULL) {
}

KinWavSed_OL::~KinWavSed_OL(void) {
Release1DArray(m_Ctrans);
Release1DArray(m_DETOverland);
Release1DArray(m_SedDep);
Release1DArray(m_SedToChannel);
Release1DArray(m_Sed_kg);
Release1DArray(m_Vol);
Release1DArray(m_V);
Release1DArray(m_Qsn);
Release1DArray(m_ChV);
Release1DArray(m_QV);
Release1DArray(m_fract);
}

void KinWavSed_OL::Get1DData(const char *key, int *n, float **data) {
*n = m_nCells;
string s(key);
if (StringMatch(s, VAR_OL_DET[0])) {
*data = m_DETOverland;
} else if (StringMatch(s, VAR_SED_DEP[0])) {
*data = m_SedDep;
} else if (StringMatch(s, VAR_SED_FLOW[0])) {
*data = m_Sed_kg;
} else if (StringMatch(s, VAR_SED_FLUX[0])) {
*data = m_Qsn;
} else if (StringMatch(s, VAR_SED_TO_CH[0])) {
*data = m_SedToChannel;
}
else {
throw ModelException(M_KINWAVSED_OL[0], "Get1DData",
"Result " + s + " does not exist in current module. Please contact the module developer.");
}

}

void KinWavSed_OL::Set1DData(const char *key, int nRows, float *data) {
string s(key);

CheckInputSize(key, nRows);

if (StringMatch(s, VAR_SLOPE[0])) { m_Slope = data; }
else if (StringMatch(s, VAR_MANNING[0])) { m_ManningN = data; }
else if (StringMatch(s, VAR_STREAM_LINK[0])) { m_streamLink = data; }
else if (StringMatch(s, VAR_USLE_K[0])) { m_USLE_K = data; }
else if (StringMatch(s, VAR_USLE_C[0])) { m_USLE_C = data; }
else if (StringMatch(s, VAR_CHWIDTH[0])) {
m_chWidth = data;
} else if (StringMatch(s, VAR_SURU[0])) { m_WH = data; }
else if (StringMatch(s, "D_QOverland")) { m_Qkin = data; }
else if (StringMatch(s, "D_DETSplash")) { m_DETSplash = data; }
else if (StringMatch(s, "D_FlowWidth")) { m_FlowWidth = data; }
else {
throw ModelException(M_KINWAVSED_OL[0], "Set1DData", "Parameter " + s +
" does not exist.");
}
}

void KinWavSed_OL::Set2DData(const char *key, int nrows, int ncols, float **data) {
string sk(key);
if (StringMatch(sk, Tag_ROUTING_LAYERS[0])) {
m_routingLayers = data;
m_nLayers = nrows;
} else if (StringMatch(sk, Tag_FLOWIN_INDEX[0])) {
m_flowInIndex = data;
} else {
throw ModelException(M_KINWAVSED_OL[0], "Set2DData", "Parameter " + sk
+ " does not exist.");
}
}

void KinWavSed_OL::SetValue(const char *key, float data) {
string s(key);
if (StringMatch(s, Tag_CellWidth[0])) { m_CellWidth = data; }
else if (StringMatch(s, Tag_CellSize[0])) { m_nCells = int(data); }
else if (StringMatch(s, Tag_HillSlopeTimeStep[0])) { m_TimeStep = data; }
else if (StringMatch(s, VAR_OL_SED_ECO1[0])) { m_eco1 = data; }
else if (StringMatch(s, VAR_OL_SED_ECO2[0])) { m_eco2 = data; }
else if (StringMatch(s, VAR_OL_SED_CCOE[0])) { m_Ccoe = data; }
else {
throw ModelException(M_KINWAVSED_OL[0], "SetValue", "Parameter " + s +
" does not exist in current module.");
}
}

bool KinWavSed_OL::CheckInputData() {
if (m_routingLayers == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "The parameter: routingLayers has not been set.");
}
if (m_flowInIndex == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "The parameter: flow in index has not been set.");
}
if (m_date < 0) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "You have not set the time.");
return false;
}
if (m_CellWidth <= 0) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "The cell width can not be less than zero.");
return false;
}
if (m_nCells <= 0) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "The cell number can not be less than zero.");
return false;
}
if (m_TimeStep < 0) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "You have not set the time step.");
return false;
}
if (m_eco1 < 0) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "You have not set the calibration coefficient 1.");
return false;
}
if (m_eco2 < 0) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "You have not set the calibration coefficient 2.");
return false;
}
if (m_Ccoe < 0) {
throw ModelException("Soil_DET", "CheckInputData",
"You have not set the calibration coefficient of overland erosion.");
return false;
}
if (m_USLE_K == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "You have not set the USLE K (Erosion factor).");
return false;
}
if (m_USLE_C == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "The parameter of USLE_C can not be NULL.");
return false;
}
if (m_Slope == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "The slope��%��can not be NULL.");
return false;
}
if (m_DETSplash == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData",
"The distribution of splash detachment can not be NULL.");
return false;
}
if (m_chWidth == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "Channel width can not be NULL.");
return false;
}
if (m_WH == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData",
"The depth of the surface water layer can not be NULL.");
return false;
}
if (m_Qkin == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "The kinematic wave flow can not be NULL.");
return false;
}
if (m_FlowWidth == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "The flow width can not be NULL.");
return false;
}
if (m_ManningN == NULL) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputData", "The Manning N can not be NULL.");
return false;
}
return true;
}

bool KinWavSed_OL::CheckInputSize(const char *key, int n) {
if (n <= 0) {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputSize",
"Input data for " + string(key) + " is invalid. The size could not be less than zero.");
return false;
}
if (m_nCells != n) {
if (m_nCells <= 0) { m_nCells = n; }
else {
throw ModelException(M_KINWAVSED_OL[0], "CheckInputSize", "Input data for " + string(key) +
" is invalid. All the input data should have same size.");
return false;
}
}

return true;
}

void KinWavSed_OL::initial() {
if (m_SedDep == NULL) {
m_SedDep = new float[m_nCells];
}
if (m_Sed_kg == NULL) {
m_Sed_kg = new float[m_nCells];
}
if (m_SedToChannel == NULL) {
m_SedToChannel = new float[m_nCells];
}
if (m_Qsn == NULL) {
m_Qsn = new float[m_nCells];
}
if (m_Vol == NULL) {
m_Vol = new float[m_nCells];
m_V = new float[m_nCells];
m_QV = new float[m_nCells];
m_Ctrans = new float[m_nCells];
m_DETOverland = new float[m_nCells];
m_ChV = new float[m_nCells];
m_fract = new float[m_nCells];
for (int i = 0; i < m_nCells; i++) {
m_Ctrans[i] = 0.0f;
m_SedDep[i] = 0.0f;
m_Sed_kg[i] = 0.0f;
m_SedToChannel[i] = 0.0f;
m_Qsn[i] = 0.0f;
m_Vol[i] = 0.0f;
m_V[i] = 0.0f;
m_QV[i] = 0.0f; 
m_DETOverland[i] = 0.0f;
m_ChV[i] = 0.0f;
m_fract[i] = 0.0f;
}
}

CalcuVelocityOverlandFlow();
WaterVolumeCalc();
}

void KinWavSed_OL::CalcuVelocityOverlandFlow() {
const float beta = 0.6f;
float Perim, R, S, n;

for (int i = 0; i < m_nCells; i++) {
if (m_WH[i] > 0.0001f) {
Perim = 2 * m_WH[i] / 1000 + m_FlowWidth[i];    
if (Perim > 0) {
R = m_WH[i] / 1000 * m_FlowWidth[i] / Perim;
} else {
R = 0.0f;
}
S = sin(atan(m_Slope[i]));   
S = Max(0.001f, S);
n = m_ManningN[i];  

m_V[i] = CalPow(R, _2div3) * CalSqrt(S) / n;
} else {
m_V[i] = 0;
}

float areaInter = m_WH[i] / 1000 + m_FlowWidth[i];
if (areaInter != 0) {
m_QV[i] = m_Qkin[i] / areaInter;
} else {
m_QV[i] = 0;
}

}
}

void KinWavSed_OL::GetTransportCapacity(int id) {
float q, S0, K;
q = m_V[id] * m_WH[id] *
60.f;   
float s = Max(0.001f, m_Slope[id]);
S0 = sin(atan(s));
K = m_USLE_K[id];
float threadhold = 0.046f;
if (q <= 0.f) {
m_Ctrans[id] = 0.f;
} else {
if (q < threadhold) {
m_Ctrans[id] = m_eco1 * K * S0 * CalSqrt(q);   
} else {
m_Ctrans[id] = m_eco2 * K * S0 * Power(q, 2.0f);
}
m_Ctrans[id] = m_Ctrans[id] / q;
}
}

void KinWavSed_OL::GetSedimentInFlow(int id) {
float TC, Df, Dsp, Deposition, concentration, Vol;
GetTransportCapacity(id);
TC = m_Ctrans[id];             
CalcuFlowDetachment(id);
Df = m_DETOverland[id];       
Dsp = m_DETSplash[id];       

m_Sed_kg[id] += (Df + Dsp);   
Vol = m_Vol[id];
if (Vol > 0) {
concentration = m_Sed_kg[id] / Vol;    
} else {
concentration = 0;
}
Deposition = Max(concentration - TC, 0.0f);   
if (Deposition > 0) {
m_Sed_kg[id] = TC * Vol;
}
m_SedDep[id] = Deposition * Vol; 
}


void KinWavSed_OL::MaxConcentration(float watvol, float sedvol, int id) {
float conc = (watvol > m_CellWidth * m_CellWidth * 1e-6 ? sedvol / watvol : 0);
if (conc > 848) {
m_SedDep[id] += Max(0.f, sedvol - 848 * watvol);
conc = 848;
}
m_Sed_kg[id] = conc * watvol;

}

void KinWavSed_OL::WaterVolumeCalc() {
float slope, DX, wh;
for (int i = 0; i < m_nCells; i++) {
slope = atan(m_Slope[i]);
DX = m_CellWidth / cos(slope);
wh = m_WH[i] / 1000;  
m_Vol[i] = DX * m_CellWidth * wh;  
}
}

void KinWavSed_OL::CalcuFlowDetachment(int i)  
{
float s = Max(0.001f, m_Slope[i]);
float S0 = sin(atan(s));
float waterdepth = m_WH[i] / 1000.f;   

float Df, waterden, g, shearStr;      
waterden = 1000;
g = 9.8f;
shearStr = waterden * g * waterdepth * S0;
Df = m_Ccoe * m_USLE_C[i] * m_USLE_K[i] * Power(shearStr, 1.5f);

float cellareas = (m_CellWidth / cos(atan(s))) * m_CellWidth;
m_DETOverland[i] = Df * (m_TimeStep / 60) * cellareas;
}

float KinWavSed_OL::SedToChannel(int ID) {
float fractiontochannel = 0.0f;
if (m_chWidth[ID] > 0) {
float tem = m_ChV[ID] * m_TimeStep;
fractiontochannel = 2 * tem / (m_CellWidth - m_chWidth[ID]);
fractiontochannel = Min(fractiontochannel, 1.0f);
}
float sedtoch = fractiontochannel * m_Sed_kg[ID];
m_Sed_kg[ID] -= sedtoch;
m_fract[ID] = fractiontochannel;

return sedtoch;
}

float KinWavSed_OL::simpleSedCalc(float Qn, float Qin, float Sin, float dt, float vol, float sed) {
float Qsn = 0;
float totsed = sed + Sin * dt;  
float totwater = vol + Qin * dt;   
if (totwater <= 1e-10) {
return (Qsn);
}
Qsn = Min(totsed / dt, Qn * totsed / totwater);
return (Qsn); 

}

float KinWavSed_OL::complexSedCalc(float Qj1i1, float Qj1i, float Qji1, float Sj1i, float Sji1, float alpha, float dt,
float dx) {
float Sj1i1, Cavg, Qavg, aQb, abQb_1, A, B, C, s = 0.f;
const float beta = 0.6f;

if (Qj1i1 < 1e-6) {
return (0);
}

Qavg = 0.5f * (Qji1 + Qj1i);
if (Qavg <= 1e-6) {
return (0);
}

Cavg = (Sj1i + Sji1) / (Qj1i + Qji1);
aQb = alpha * Power(Qavg, beta);
abQb_1 = alpha * beta * Power(Qavg, beta - 1);

A = dt * Sj1i;
B = -dx * Cavg * abQb_1 * (Qj1i1 - Qji1);
C = (Qji1 <= 1e-6 ? 0 : dx * aQb * Sji1 / Qji1);
if (Qj1i1 > 1e-6) {
Sj1i1 = (dx * dt * s + A + C + B) / (dt + dx * aQb / Qj1i1);    
} else {
Sj1i1 = 0;
}
Sj1i1 = Max(0.f, Sj1i1);
return Sj1i1;
}

void KinWavSed_OL::OverlandflowSedRouting(int id) {
float flowwidth = m_FlowWidth[id];
float Sin = 0.0f;
float Qin = 0.0f;
for (int k = 1; k <= (int) m_flowInIndex[id][0]; ++k) {
int flowInID = (int) m_flowInIndex[id][k];
Qin += m_Qkin[flowInID];   
Sin += m_Qsn[flowInID];        
}

if (m_streamLink[id] >= 0 && flowwidth <= 0) {
m_SedToChannel[id] = Sin * m_TimeStep;
return;
}

float WtVol = m_Vol[id];
GetSedimentInFlow(id);
m_Qsn[id] = simpleSedCalc(m_Qkin[id], Qin, Sin, m_TimeStep, WtVol, m_Sed_kg[id]);
float tem = Sin + m_Sed_kg[id] / m_TimeStep;
m_Qsn[id] = Min(m_Qsn[id], tem);
tem = Sin * m_TimeStep + m_Sed_kg[id] - m_Qsn[id] * m_TimeStep;
m_Sed_kg[id] = Max(0.0f, tem);
m_SedToChannel[id] = SedToChannel(id);

}

int KinWavSed_OL::Execute() {
CheckInputData();

initial();


for (int iLayer = 0; iLayer < m_nLayers; ++iLayer) {
int nCells = (int) m_routingLayers[iLayer][0];
#pragma omp parallel for
for (int iCell = 1; iCell <= nCells; ++iCell) {
int id = (int) m_routingLayers[iLayer][iCell];
OverlandflowSedRouting(id);
}
}

return 0;
}