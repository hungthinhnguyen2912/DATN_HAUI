import 'package:flutter/material.dart';
import 'package:mobile_app/controllers/avatar_controller.dart';
import 'package:get/get.dart';

import '../../App_Color.dart';
import '../classification/classification_page.dart';
import '../history/history_page.dart';
import '../home/home_page.dart';
import '../setting/setting_page.dart';

class myBottomNavBar extends StatefulWidget {
  const myBottomNavBar({super.key});

  @override
  State<myBottomNavBar> createState() => _myBottomNavBarState();
}

class _myBottomNavBarState extends State<myBottomNavBar> {
  AvatarController avatarController = Get.put(AvatarController());
  int _selectedIndex = 0;
  final List<Widget> _screens = [
    HomePage(),
    ClassificationPage(),
    HistoryPage(),
    SettingPage(),
  ];

  @override
  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        backgroundColor: AppColors.green,
        items: [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: "Home"),
          BottomNavigationBarItem(
            icon: Icon(Icons.search),
            label: "Classification",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.access_time_rounded),
            label: "History",
          ),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: "Setting"),
        ],
        currentIndex: _selectedIndex,
        elevation: 0,
        selectedItemColor: Colors.green,
        unselectedItemColor: AppColors.gray_login_text,
        onTap: _onItemTapped,
      ),
    );
  }
}
