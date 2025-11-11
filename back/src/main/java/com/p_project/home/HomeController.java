package com.p_project.home;

import com.p_project.oauth2.CustomOAuth2User;
import com.p_project.user.UserEntity;
import com.p_project.user.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.Optional;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/home")
public class HomeController {

    private final HomeService homeService;
    private final UserRepository userRepository;

    @GetMapping("/{userId}")
    public ResponseEntity<HomeDTO> getHome(@PathVariable Long userId){
        log.info("in HomeController: getHome");
        HomeDTO response = homeService.getHome(userId);

        return ResponseEntity.ok(response);
    }

    // main
    @GetMapping("/main")
    @ResponseBody
    public String mainAPI(){

        return "main route";
    }

    @GetMapping("/test")
    @ResponseBody
    public ResponseEntity<Long> testAPI(Authentication authentication){
        log.info(">>> [Controller] 진입 성공");
        log.info("token = {}", authentication.getPrincipal());

        CustomOAuth2User customUser = (CustomOAuth2User) authentication.getPrincipal();
        String email = customUser.getEmail();

        Optional<UserEntity> userDTO = userRepository.findByEmail(email);

        Long userId = userDTO.get().getId();
        log.info(String.valueOf(userId));

        return ResponseEntity.ok(userId);
    }


}
