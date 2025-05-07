import 'package:get/get.dart';
import 'package:mobile_app/routes/route_name.dart';
import 'package:mobile_app/views/Widget/bottom_nav_bar.dart';
import 'package:mobile_app/views/auth/splash_screen.dart';
import 'package:mobile_app/views/setting/items_setting_page/change_pass.dart';
import 'package:mobile_app/views/setting/items_setting_page/confirm_page.dart';
import 'package:mobile_app/views/setting/items_setting_page/edit_profile_page.dart';

import '../views/auth/auth_phone/verify_code.dart';

class AppRoute {
  static final page = [
    GetPage(name: RouteName.initial, page: () => SplashScreen()),
    GetPage(name: RouteName.home, page: () => myBottomNavBar()),
    GetPage(name: RouteName.auth, page: () => SplashScreen()),
    GetPage(name: RouteName.editProfile, page: () => EditProfilePage()),
    GetPage(name: RouteName.changePass, page: () => ChangePassPage()),
    GetPage(name: RouteName.confirmPass, page: () => ConfirmPage()),
    GetPage(name: RouteName.verifyCode,page: () => VerifyCode())
  ];
}
