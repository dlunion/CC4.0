
#include <cv.h>
#include <windows.h>
#include "caffe/util/pa_draw.h"

void GetStringSize(HDC hDC, const char* str, int* w, int* h)
{
	SIZE size;
	GetTextExtentPoint32A(hDC, str, strlen(str), &size);
	if(w != 0) *w = size.cx;
	if(h != 0) *h = size.cy;
}

void paDrawString(cv::Mat& _dst, const std::string& str, cv::Point org, cv::Scalar color, int fontSize, bool bold, bool italic, bool underline)
//void paDrawString(IplImage* dst, const char* str, CvPoint org, CvScalar color, int fontSize, bool italic, bool underline)
{
	IplImage ipldst = _dst;
	IplImage* dst = &ipldst;
	CV_Assert(dst != 0 && dst->depth == IPL_DEPTH_8U && (dst->nChannels == 1 || dst->nChannels == 3));

	int x, y, r, b;
	if(org.x > dst->width || org.y > dst->height) return;

	LOGFONTA lf;
	lf.lfHeight         = - fontSize ;
	lf.lfWidth          = 0 ;
	lf.lfEscapement     = 0 ;
	lf.lfOrientation    = 0 ;
	lf.lfWeight			= bold ? FW_BOLD : FW_NORMAL;
	lf.lfItalic         = italic ;	//斜体
	lf.lfUnderline      = underline ;	//下划线
	lf.lfStrikeOut       = 0 ;
	lf.lfCharSet        = DEFAULT_CHARSET ;
	lf.lfOutPrecision    = 0 ;
	lf.lfClipPrecision    = 0 ;
	lf.lfQuality         = PROOF_QUALITY ;
	lf.lfPitchAndFamily  = 0 ;
	strcpy (lf.lfFaceName, "宋体" );

	HFONT hf = CreateFontIndirectA(&lf);
	HDC hDC = CreateCompatibleDC(0);
	HFONT hOldFont = (HFONT)SelectObject(hDC, hf);

	int strBaseW = 0, strBaseH = 0;
	int singleRow = 0;
	char buf[3000];
	strcpy(buf, str.c_str());

	//处理多行
	{
		int nnh = 0;
		int cw, ch;
		const char* ln = strtok(buf, "\n");
		while(ln != 0)
		{
			GetStringSize(hDC, ln, &cw, &ch);
			strBaseW = max(strBaseW, cw);
			strBaseH = max(strBaseH, ch);

			ln = strtok(0, "\n");
			nnh++;
		}
		singleRow = strBaseH;
		strBaseH *= nnh;
	}

	int centerx = 0;
	int centery = 0;
	if (org.x < ORG_Center*0.5){
		org.x = (dst->width - strBaseW) * 0.5 + (org.x - ORG_Center);
		centerx = 1;
	}

	if (org.y < ORG_Center*0.5){
		org.y = (dst->height - strBaseH) * 0.5 + (org.y - ORG_Center);
		centery = 1;
	}
	
	x = org.x < 0 ? -org.x : 0;
	y = org.y < 0 ? -org.y : 0;

	if(org.x + strBaseW < 0 || org.y + strBaseH < 0)
	{
		SelectObject(hDC, hOldFont);
		DeleteObject(hf);
		DeleteObject(hDC);
		return;
	}

	r = org.x + strBaseW > dst->width ? dst->width - org.x - 1 : strBaseW - 1;	
	b = org.y + strBaseH > dst->height ? dst->height - org.y - 1 : strBaseH - 1;
	org.x = org.x < 0 ? 0 : org.x;
	org.y = org.y < 0 ? 0 : org.y;

	BITMAPINFO bmp = {0};
	BITMAPINFOHEADER& bih = bmp.bmiHeader;
	int strDrawLineStep = strBaseW * 3 % 4 == 0 ? strBaseW * 3 : (strBaseW * 3 + 4 - ((strBaseW * 3) % 4));

	bih.biSize=sizeof(BITMAPINFOHEADER);
	bih.biWidth=strBaseW;
	bih.biHeight=strBaseH;
	bih.biPlanes=1;
	bih.biBitCount=24;
	bih.biCompression=BI_RGB;
	bih.biSizeImage=strBaseH * strDrawLineStep;
	bih.biClrUsed=0;
	bih.biClrImportant=0;

	void* pDibData = 0;
	HBITMAP hBmp = CreateDIBSection(hDC, &bmp, DIB_RGB_COLORS, &pDibData, 0, 0);
	if (pDibData == 0){
		SelectObject(hDC, hOldFont);
		DeleteObject(hf);
		DeleteObject(hDC);
		return;
	}

	CV_Assert(pDibData != 0);
	HBITMAP hOldBmp = (HBITMAP)SelectObject(hDC, hBmp);

	//color.val[2], color.val[1], color.val[0]
	SetTextColor(hDC, RGB(255, 255, 255));
	SetBkColor(hDC, 0);
	//SetStretchBltMode(hDC, COLORONCOLOR);

	strcpy(buf, str.c_str());
	const char* ln = strtok(buf, "\n");
	int outTextY = 0;
	while(ln != 0)
	{
		if (centerx){
			int cw, ch;
			GetStringSize(hDC, ln, &cw, &ch);
			TextOutA(hDC, (strBaseW - cw) * 0.5, outTextY, ln, strlen(ln));
		}
		else{
			TextOutA(hDC, 0, outTextY, ln, strlen(ln));
		}
		outTextY += singleRow;
		ln = strtok(0, "\n");
	}

	unsigned char* pImg = (unsigned char*)dst->imageData + org.x * dst->nChannels + org.y * dst->widthStep;
	unsigned char* pStr = (unsigned char*)pDibData + x * 3;
	for(int tty = y; tty <= b; ++tty)
	{
		unsigned char* subImg = pImg + (tty - y) * dst->widthStep;
		unsigned char* subStr = pStr + (strBaseH - tty - 1) * strDrawLineStep;
		for (int ttx = x; ttx <= r; ++ttx)
		{
			for (int n = 0; n < dst->nChannels; ++n){
				double vtxt = subStr[n] / 255.0;
				int cvv =  vtxt * color.val[n] + (1 - vtxt) * subImg[n];
				subImg[n] = cvv > 255 ? 255 : (cvv < 0 ? 0 : cvv);
			}

			subStr += 3;
			subImg += dst->nChannels;
		}
	}

	SelectObject(hDC, hOldBmp);
	SelectObject(hDC, hOldFont);
	DeleteObject(hf);
	DeleteObject(hBmp);
	DeleteDC(hDC);
}