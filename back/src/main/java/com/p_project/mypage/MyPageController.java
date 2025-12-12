package com.p_project.mypage;

import com.p_project.oauth2.CustomOAuth2User;
import com.p_project.profile.ProfileDTO;
import com.p_project.user.UserEntity;
import com.p_project.user.UserService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.tomcat.util.http.fileupload.FileUploadException;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/mypage")
public class MyPageController {

    private final MyPageService mypageService;
    private final UserService userService;

    @PostMapping(path = "/profile", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ProfileDTO> updateProfile(
            Authentication auth,
            @RequestPart("file") MultipartFile file) throws FileUploadException {
        CustomOAuth2User principal = (CustomOAuth2User) auth.getPrincipal();
        ProfileDTO result = mypageService.updateProfile(principal.getUserId(), file);
        return ResponseEntity.ok(result);
    }

    @GetMapping
    public ResponseEntity<MyPageDTO> getMyPage(Authentication auth) {
        CustomOAuth2User principal = (CustomOAuth2User) auth.getPrincipal();
        MyPageDTO result = mypageService.getMyPage(principal.getUserId());
        return ResponseEntity.ok(result);
    }

    @PostMapping(path = "/update")
    public ResponseEntity<String> updateMyPage(
            Authentication auth,
            @RequestBody MyPageUpdateDTO myPageDTO) {

        CustomOAuth2User principal = (CustomOAuth2User) auth.getPrincipal();
        Long userId = principal.getUserId();
        myPageDTO.setUserId(userId);

        UserEntity user = userService.findById(userId)
                .orElseThrow(() -> new IllegalStateException("존재하지 않는 사용자"));

        if (userService.existsEmailExceptUser(
                myPageDTO.getEmail(), myPageDTO.getUserId())) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body("이미 사용 중인 이메일입니다.");
        }

        if (userService.existsNicknameExceptUser(
                myPageDTO.getNickName(), myPageDTO.getUserId())) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body("이미 사용 중인 닉네임입니다.");
        }

        mypageService.updateMyPage(myPageDTO);
        return ResponseEntity.ok("마이페이지 업데이트 성공");
    }

}
