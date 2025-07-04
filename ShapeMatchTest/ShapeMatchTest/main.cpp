// ShapeMatchTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "ShapeMatch.h"
//#include <time.h>

void DrawContours(cv::Mat& source, cv::Point Result, CvPoint* Contours, int ContoursSize, cv::Scalar color, int lineWidth)
{
        cv::Point point;
        int x = Result.x;
        int y = Result.y;
        cv::line(source, cv::Point(x , y-5), cv::Point(x, y+5), CV_RGB( 0, 0, 255 ), lineWidth);
        cv::line(source, cv::Point(x-5 , y), cv::Point(x+5, y), CV_RGB( 0, 0, 255 ), lineWidth);
        for (int i = 0; i < ContoursSize; i++)
        {
                point.x = Contours[i].x + x;
                point.y = Contours[i].y + y;
                cv::line(source, point, point, color, lineWidth);
        }
}

void DrawContours(cv::Mat& source, CvPoint* Contours, int ContoursSize, cv::Scalar color, int lineWidth)
{
        cv::Point point;
        int x = source.cols / 2;
        int y = source.rows / 2;
        cv::line(source, cv::Point(x , y-5), cv::Point(x, y+5), CV_RGB( 0, 0, 255 ), lineWidth);
        cv::line(source, cv::Point(x-5 , y), cv::Point(x+5, y), CV_RGB( 0, 0, 255 ), lineWidth);
        for (int i = 0; i < ContoursSize; i++)
        {
                point.x = Contours[i].x;
                point.y = Contours[i].y;
                cv::line(source, point, point, color, lineWidth);
        }
}


int main(int argc, char** argv)
{
        CShapeMatch SM;
        cv::Mat templateImage = cv::imread("ShapeMatchTest/TestImage/1.bmp", cv::IMREAD_UNCHANGED);
        if (templateImage.empty())
        {
                cout<< " 图片加载失败！\n";
                system("pause");
                return 0;
        }
        cv::Size templateSize = templateImage.size();
        cv::Mat grayTemplateImg(templateSize, CV_8UC1);

        int dim = min(templateImage.cols, templateImage.rows);
	int numoctaves = (int) (log((double) dim) / log(2.0)) - 2;    //金字塔阶数  
	numoctaves = min(numoctaves, 7);		//限定金字塔的阶梯数  

	/* Convert color image to gray image. */
        if(templateImage.channels() == 3)
        {
                cv::cvtColor(templateImage, grayTemplateImg, cv::COLOR_RGB2GRAY);
        }
        else
        {
                templateImage.copyTo(grayTemplateImg);
        }

	/* Set model parameter */
	shape_model ModelID;
	ModelID.m_AngleStart		=  0;										//起始角度
	ModelID.m_AngleStop  	= 0;										//终止角度
	ModelID.m_AngleStep		= 1;										//角度步长
	ModelID.m_Contrast			= 80;									//高阈值
	ModelID.m_MinContrast	= 30;									//低阈值
	ModelID.m_NumLevels		= 3;										//金字塔级数
	ModelID.m_Granularity     = 1;									    //颗粒度
        ModelID.m_ImageWidth   = grayTemplateImg.cols;
        ModelID.m_ImageHeight  = grayTemplateImg.rows;

        /* Train shape model and draw contours in  model image.*/
        edge_list EdgeList;
        EdgeList.EdgePiont = (CvPoint *) malloc(grayTemplateImg.cols * grayTemplateImg.rows * sizeof(CvPoint));
        IplImage* grayTemplateImg_ipl = cvCreateImageHeader(cvSize(grayTemplateImg.cols, grayTemplateImg.rows), IPL_DEPTH_8U, grayTemplateImg.channels());
        grayTemplateImg_ipl->imageData = reinterpret_cast<char*>(grayTemplateImg.data);
        grayTemplateImg_ipl->widthStep = grayTemplateImg.step;
        SM.train_shape_model(grayTemplateImg_ipl, ModelID.m_Contrast, ModelID.m_MinContrast, ModelID.m_Granularity, &EdgeList);
        cvReleaseImageHeader(&grayTemplateImg_ipl);
        DrawContours(templateImage, EdgeList.EdgePiont, EdgeList.ListSize , CV_RGB( 255, 0, 0 ),1);
        // Removed GUI window for headless execution
        // cv::namedWindow("Template", cv::WINDOW_AUTOSIZE );
        // cv::imshow("Template", templateImage);

        SM.initial_shape_model(&ModelID, grayTemplateImg.cols, grayTemplateImg.rows, EdgeList.ListSize);
	free(EdgeList.EdgePiont);

	cout<< "\n Search Model Program\n"; 
	cout<< " ------------------------------------\n";
	cout<< " 角度范围：" <<ModelID.m_AngleStart <<"°~ "<<ModelID.m_AngleStop<<"°\n";

	/* Create shape model file*/
        IplImage* grayTemplateImg_ipl2 = cvCreateImageHeader(cvSize(grayTemplateImg.cols, grayTemplateImg.rows), IPL_DEPTH_8U, grayTemplateImg.channels());
        grayTemplateImg_ipl2->imageData = reinterpret_cast<char*>(grayTemplateImg.data);
        grayTemplateImg_ipl2->widthStep = grayTemplateImg.step;
        clock_t start_time = clock();
        bool IsInial = SM.create_shape_model(grayTemplateImg_ipl2, &ModelID);
        cvReleaseImageHeader(&grayTemplateImg_ipl2);
        clock_t finish_time = clock();

	double total_time = (double)(finish_time-start_time)/CLOCKS_PER_SEC;
	cout<< " ------------------------------------\n";
	cout<<" Create Time = "<<total_time*1000<<"ms\n";

        /* Search  model */
        cv::Mat searchImage = cv::imread("ShapeMatchTest/TestImage/a.bmp", cv::IMREAD_UNCHANGED);
        if (searchImage.empty())
        {
                cout<< " 图片加载失败！\n";
                system("pause");
                return 0;
        }
        cv::Size searchSize = searchImage.size();
        cv::Mat graySearchImg(searchSize, CV_8UC1);

	/* Convert color image to gray image. */ 
        if(searchImage.channels() == 3)
                cv::cvtColor(searchImage, graySearchImg, cv::COLOR_RGB2GRAY);
        else
        {
                searchImage.copyTo(graySearchImg);
        }

	/* Set match parameter */
	int NumMatch		= 4;				//匹配个数
	float MinScore     = 0.7f;			//最小评分
	float Greediness   = 0.7f;			//贪婪度

	MatchResultA Result[10];
	memset(Result, 0, 10 * sizeof(MatchResultA));
	cout<< " ------------------------------------\n";
        if(IsInial)
        {
                IplImage* graySearchImg_ipl = cvCreateImageHeader(cvSize(graySearchImg.cols, graySearchImg.rows), IPL_DEPTH_8U, graySearchImg.channels());
                graySearchImg_ipl->imageData = reinterpret_cast<char*>(graySearchImg.data);
                graySearchImg_ipl->widthStep = graySearchImg.step;
                start_time = clock();
                SM.find_shape_model(graySearchImg_ipl, &ModelID, MinScore, NumMatch, Greediness, Result);
                finish_time = clock();
                cvReleaseImageHeader(&graySearchImg_ipl);
        }
	else
		printf(" Create model failed!\n");

	total_time = (double)(finish_time-start_time)/CLOCKS_PER_SEC;
	cout<<" Find Time = "<<total_time*1000<<"ms\n";
	cout<< " ------------------------------------\n";

	/* Draw contours in search image. */
	ShapeInfo *temp = ModelID.m_pShapeInfoTmpVec;
	CvPoint * Contours = temp->Coordinates;
	int count = temp->NoOfCordinates;
	temp = ModelID.m_pShapeInfoTmpVec;
	Contours = temp->Coordinates;
	count = 0;
	for (int n = 0; n <NumMatch; n++)
	{
		if (Result[n].ResultScore != 0)
		{
			for (int i = 0; i < temp[0].AngleNum; i++)
			{
				if (temp[i].Angel == Result[n].Angel)
				{
					Contours = temp[i].Coordinates;
					count = temp[i].NoOfCordinates;
					break;
				}
			}
                        printf(" Location:(%d, %d) Angle: %d Score: %.4f\n", Result[n].CenterLocX, Result[n].CenterLocY, Result[n].Angel, Result[n].ResultScore);
                        DrawContours(searchImage, cv::Point(Result[n].CenterLocX, Result[n].CenterLocY), Contours, count, CV_RGB( 0, 255, 0 ),1);
		}
	}
	SM.release_shape_model(&ModelID);
	
	//Display result
        // Removed GUI window for headless execution
        // cv::namedWindow("Search Image", cv::WINDOW_AUTOSIZE );
        // cv::imshow("Search Image", searchImage);

        // Wait and cleanup removed for no-window mode
        // cv::waitKey( 0 );
        // cv::destroyWindow("Search Image");
        // cv::destroyWindow("Template");

	return 0;
}
