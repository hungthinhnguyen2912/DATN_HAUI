import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';
import 'package:mobile_app/App_Color.dart';

class DetailVegetablePage extends StatelessWidget {
  final String name;
  final String title;
  final String detail;

  const DetailVegetablePage({
    super.key,
    required this.name,
    required this.title,
    required this.detail,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppColors.green,
        title: Text(name),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              CarouselSlider(
                items: List.generate(4, (index) {
                  return ClipRRect(
                    borderRadius: BorderRadius.circular(15),
                    child: Image.asset(
                      "assets/images/Real $name/${index + 1}.jpg",
                      width: double.infinity,
                      fit: BoxFit.cover,
                    ),
                  );
                }),
                options: CarouselOptions(
                  height: 250,
                  enlargeCenterPage: true,
                  autoPlay: true,
                  aspectRatio: 16 / 9,
                  autoPlayCurve: Curves.fastOutSlowIn,
                  enableInfiniteScroll: true,
                  autoPlayAnimationDuration: Duration(milliseconds: 800),
                ),
              ),
              SizedBox(height: 20),
              Text(
                title,
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: AppColors.green,
                ),
                textAlign: TextAlign.center,
              ),
              Container(
                padding: EdgeInsets.all(15),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(15),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 10,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: Text(
                  detail,
                  style: TextStyle(fontSize: 18, height: 1.5),
                  textAlign: TextAlign.justify,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
